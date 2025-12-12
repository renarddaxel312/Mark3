import numpy as np
import torch
import time

from functions import (
    parse_urdf,
    xyz_rpy_to_matrix,
)


def forward_kinematics_pos(urdf_info, q_deg, return_points=False):
    T = np.eye(4)
    q_rad = np.deg2rad(q_deg[: urdf_info["n_joints"]])
    points = [T[:3, 3].copy()] if return_points else None

    for entry in urdf_info["chain"]:
        T = T @ xyz_rpy_to_matrix(entry["transform"]["xyz"], entry["transform"]["rpy"])

        if entry["type"] == "revolute":
            idx = entry["joint_index"]
            theta = q_rad[idx]
            Rz = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            R_joint = np.eye(4)
            R_joint[:3, :3] = Rz
            T = T @ R_joint

        if return_points:
            points.append(T[:3, 3].copy())

    if return_points:
        return np.array(points)
    return T[:3, 3]


def numeric_jacobian_pos(urdf_info, q_deg, target_pos=None, eps_deg=1e-4, skip_first=True):
    n = urdf_info["n_joints"]
    J = np.zeros((3, n))
    base_pos = forward_kinematics_pos(urdf_info, q_deg)

    start_idx = 1 if skip_first else 0
    for i in range(start_idx, n):
        qd = np.array(q_deg, float)
        qd[i] += eps_deg
        pos_d = forward_kinematics_pos(urdf_info, qd)
        J[:, i] = (pos_d - base_pos) / eps_deg

    return J


def inverse_kinematics_pos(
    urdf_info,
    target_pos,
    q_init=None,
    max_iter=2000,
    lr=0.3,
    lam=1e-3,
    tol=1e-4,
    eps_fd=1e-4,
    n_restarts=2,
    use_base_yaw=True,
    return_stats=False,
):
    n = urdf_info["n_joints"]
    limits = urdf_info["joint_limits"]
    target_pos = np.asarray(target_pos, float)

    def clamp_angles_deg(q_deg):
        q_deg = np.asarray(q_deg, float).copy()
        for i in range(n):
            lo, hi = limits[i]  # radians
            q_deg[i] = np.clip(q_deg[i], np.rad2deg(lo), np.rad2deg(hi))
        return q_deg

    best_q = None
    best_err = np.inf
    best_stats = {"converged": False, "iterations": None, "restart": None, "best_err": None}

    q0_fixed = np.rad2deg(np.arctan2(target_pos[1], target_pos[0] + 1e-12))

    for r in range(max(n_restarts, 1)):
        if q_init is None:
            q = np.random.uniform(-45.0, 45.0, size=n)
        else:
            q = np.array(q_init, float) + np.random.normal(0.0, 5.0, size=n)
        q = clamp_angles_deg(q)

        if use_base_yaw and n > 0:
            q[0] = q0_fixed

        prev_err = np.inf
        current_lr = lr

        for it in range(max_iter):
            pos = forward_kinematics_pos(urdf_info, q)
            err = target_pos - pos
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                q_out = clamp_angles_deg(q)
                if return_stats:
                    return q_out, {"converged": True, "iterations": int(it + 1), "restart": int(r), "best_err": float(err_norm)}
                return q_out

            J = numeric_jacobian_pos(
                urdf_info,
                q,
                target_pos=None,
                eps_deg=eps_fd,
                skip_first=use_base_yaw and n > 0,
            )
            Jt = J.T
            step = Jt @ np.linalg.inv(J @ Jt + lam * np.eye(3)) @ err
            dq = current_lr * step
            q = q + dq

            q = clamp_angles_deg(q)

            if use_base_yaw and n > 0:
                q[0] = q0_fixed

            if err_norm > prev_err * 1.2:
                current_lr *= 0.5

            if err_norm < best_err:
                best_err = err_norm
                best_q = q.copy()
                best_stats = {"converged": False, "iterations": int(it + 1), "restart": int(r), "best_err": float(best_err)}
            prev_err = err_norm

    q_out = clamp_angles_deg(best_q if best_q is not None else np.zeros(n))
    if return_stats:
        return q_out, best_stats
    return q_out


def load_urdf_info_from_config(config):
    from functions import urdf  # local import to avoid cycles on load

    urdf_string = urdf(config)
    return parse_urdf(urdf_string)


def urdf_chain_to_torch(urdf_info, device="cpu"):
    chain_ops = []
    for entry in urdf_info["chain"]:
        T_fixed = xyz_rpy_to_matrix(entry["transform"]["xyz"], entry["transform"]["rpy"])
        T_fixed_t = torch.tensor(T_fixed, dtype=torch.float32, device=device)
        
        op = {
            "type": entry["type"],
            "fixed": T_fixed_t,
            "joint_index": entry.get("joint_index")
        }
        chain_ops.append(op)
    
    limits = torch.tensor(urdf_info["joint_limits"], dtype=torch.float32, device=device)
    return chain_ops, limits


def forward_kinematics_batch(chain_ops, q_deg):
    batch_size = q_deg.shape[0]
    device = q_deg.device
    
    q_rad = torch.deg2rad(q_deg)
    
    T = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    for op in chain_ops:
        T = torch.matmul(T, op["fixed"])
        
        if op["type"] == "revolute":
            idx = op["joint_index"]
            theta = q_rad[:, idx]
            
            c = torch.cos(theta)
            s = torch.sin(theta)
            z = torch.zeros_like(theta)
            o = torch.ones_like(theta)
            
            R = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            R[:, 0, 0] = c
            R[:, 0, 1] = -s
            R[:, 1, 0] = s
            R[:, 1, 1] = c
            
            T = torch.matmul(T, R)
            
    return T[:, :3, 3]


def compute_jacobian_batch(chain_ops, q_deg):
    batch_size = q_deg.shape[0]
    n_joints = q_deg.shape[1]
    
    q_in = q_deg.clone().requires_grad_(True)
    
    pos = forward_kinematics_batch(chain_ops, q_in)
    
    # We need to compute gradients of pos_x, pos_y, pos_z w.r.t q
    # We can do this efficiently by creating a grad_output vectors
    
    jacobians = torch.zeros(batch_size, 3, n_joints, device=q_deg.device)
    
    for i in range(3):
        grad_output = torch.zeros(batch_size, 3, device=q_deg.device)
        grad_output[:, i] = 1.0
        
        grads = torch.autograd.grad(
            outputs=pos,
            inputs=q_in,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]
        
        if grads is None:
            grads = torch.zeros_like(q_in)

        jacobians[:, i, :] = grads
        
    return jacobians


def inverse_kinematics_batch(
    urdf_info,
    targets,
    q_init=None,
    max_iter=100,
    lr=0.25,
    lam=1e-3,
    tol=5e-3,
    use_base_yaw=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    log_interval=50,
    return_stats=False,
):
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
    else:
        targets = targets.to(device)
        
    batch_size = targets.shape[0]
    n_joints = urdf_info["n_joints"]
    
    chain_ops, limits = urdf_chain_to_torch(urdf_info, device=device)
    
    lows = torch.rad2deg(limits[:, 0])
    highs = torch.rad2deg(limits[:, 1])
    
    if q_init is None:
        q = (torch.rand(batch_size, n_joints, device=device) * 90.0) - 45.0
    else:
        if not isinstance(q_init, torch.Tensor):
            q = torch.tensor(q_init, dtype=torch.float32, device=device)
        else:
            q = q_init.to(device)
        q = q + torch.randn_like(q) * 5.0

    q = torch.max(torch.min(q, highs), lows)
        
    if use_base_yaw and n_joints > 0:
        q[:, 0] = torch.rad2deg(torch.atan2(targets[:, 1], targets[:, 0] + 1e-12))
        
    lr_vec = torch.full((batch_size, 1), float(lr), dtype=torch.float32, device=device)
    prev_err_norm = torch.full((batch_size,), float("inf"), dtype=torch.float32, device=device)
    first_converged_iter = torch.full((batch_size,), -1, dtype=torch.int32, device=device)

    for it in range(max_iter):
        pos = forward_kinematics_batch(chain_ops, q)
        err = targets - pos
        err_norm = torch.norm(err, dim=1)
        if verbose and (it % log_interval == 0 or it == max_iter - 1):
            mean_err = err_norm.mean().item()
            max_err = err_norm.max().item()
            print(f"[IK batch] iter {it:04d}/{max_iter} | mean err {mean_err:.6f} | max err {max_err:.6f}")

        if return_stats:
            newly = (first_converged_iter < 0) & (err_norm <= tol)
            first_converged_iter[newly] = int(it + 1)
        

        
        J = compute_jacobian_batch(chain_ops, q) # (B, 3, N)
        Jt = J.transpose(1, 2) # (B, N, 3)
        

        # Compute JJ^T
        JJt = torch.matmul(J, Jt) # (B, 3, 3)
        
        # Add damping
        eye = torch.eye(3, device=device).unsqueeze(0)
        damped = JJt + lam * eye
        
        # Solve linear system 

        err_unsqueezed = err.unsqueeze(2)
        
        try:
            rhs = torch.linalg.solve(damped, err_unsqueezed) # (B, 3, 1)
        except RuntimeError:
        
            damped = damped + 1e-2 * eye
            rhs = torch.linalg.solve(damped, err_unsqueezed)

        dq = torch.matmul(Jt, rhs).squeeze(2) # (B, N, 1) -> (B, N)
        
        if use_base_yaw and n_joints > 0:
            dq[:, 0] = 0.0
        
        diverging = err_norm > (prev_err_norm * 1.2)
        if diverging.any():
            lr_vec[diverging.unsqueeze(1)] *= 0.5
        prev_err_norm = err_norm

        q = q + lr_vec * dq
        
        q = torch.max(torch.min(q, highs), lows)
        
        if use_base_yaw and n_joints > 0:
            q[:, 0] = torch.rad2deg(torch.atan2(targets[:, 1], targets[:, 0] + 1e-12))
            
    # Final check
    final_pos = forward_kinematics_batch(chain_ops, q)
    final_err_norm = torch.norm(targets - final_pos, dim=1)
    
    q_np = q.cpu().detach().numpy()
    err_np = final_err_norm.cpu().detach().numpy()
    if return_stats:
        stats = {
            "first_converged_iter": first_converged_iter.cpu().numpy(),
            "final_lr": lr_vec.squeeze(1).cpu().numpy(),
        }
        return q_np, err_np, stats
    return q_np, err_np
