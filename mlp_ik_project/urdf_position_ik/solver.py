import numpy as np
import torch
import time

from functions import (
    parse_urdf,
    xyz_rpy_to_matrix,
)


def forward_kinematics_pos(urdf_info, q_deg, return_points=False):
    """
    Forward kinematics returning end-effector position (meters).
    
    If return_points is True, returns an array of link frame positions along the chain.
    Revolute joints are assumed to rotate about their local Z axis.
    """
    T = np.eye(4)
    q_rad = np.deg2rad(q_deg[: urdf_info["n_joints"]])
    points = [T[:3, 3].copy()] if return_points else None

    for entry in urdf_info["chain"]:
        # Apply fixed transform
        T = T @ xyz_rpy_to_matrix(entry["transform"]["xyz"], entry["transform"]["rpy"])

        # Apply revolute rotation about local Z
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
    """
    Node-style finite-difference 3xN Jacobian of end-effector position w.r.t. joint angles (deg).
    - Default eps matches IKsolverNode (very small).
    - Default skip_first=True to exclude joint0 (base-yaw heuristic).
    """
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
    """
    Position-only IK using node-style damped least squares.
    
    Args:
        urdf_info: output of parse_urdf
        target_pos: iterable[3], meters
        q_init: optional initial guess in degrees (len = n_joints)
    Returns:
        q solution (degrees)
    """
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
            # Node behavior: small uniform box then clamp
            q = np.random.uniform(-45.0, 45.0, size=n)
        else:
            # Node behavior: perturb provided init
            q = np.array(q_init, float) + np.random.normal(0.0, 5.0, size=n)
        q = clamp_angles_deg(q)

        # Base yaw heuristic: align joint 0 with target projection
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

            # Enforce base yaw heuristic after clamp
            if use_base_yaw and n > 0:
                q[0] = q0_fixed

            # Adapt learning rate if diverging
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
    """
    Convenience: build URDF string from a joint-type config and parse it.
    """
    from functions import urdf  # local import to avoid cycles on load

    urdf_string = urdf(config)
    return parse_urdf(urdf_string)


# -----------------------------------------------------------------------------
# PyTorch Batch Implementation
# -----------------------------------------------------------------------------

def urdf_chain_to_torch(urdf_info, device="cpu"):
    """
    Convert URDF chain info to PyTorch tensors for batch processing.
    """
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
    """
    Batch FK.
    q_deg: (batch_size, n_joints) degrees
    Returns: (batch_size, 3) end-effector positions
    """
    batch_size = q_deg.shape[0]
    device = q_deg.device
    
    # Convert to radians
    q_rad = torch.deg2rad(q_deg)
    
    # Start with Identity
    T = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    for op in chain_ops:
        # Apply fixed transform
        T = torch.matmul(T, op["fixed"])
        
        if op["type"] == "revolute":
            idx = op["joint_index"]
            theta = q_rad[:, idx]
            
            c = torch.cos(theta)
            s = torch.sin(theta)
            z = torch.zeros_like(theta)
            o = torch.ones_like(theta)
            
            # Rz matrix
            # [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            R = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            R[:, 0, 0] = c
            R[:, 0, 1] = -s
            R[:, 1, 0] = s
            R[:, 1, 1] = c
            
            T = torch.matmul(T, R)
            
    return T[:, :3, 3]


def compute_jacobian_batch(chain_ops, q_deg):
    """
    Compute Jacobian J (3xN) for each item in batch via autograd.
    Returns: (batch_size, 3, n_joints)
    """
    batch_size = q_deg.shape[0]
    n_joints = q_deg.shape[1]
    
    # Enable grad for q
    q_in = q_deg.clone().requires_grad_(True)
    
    # Forward pass
    pos = forward_kinematics_batch(chain_ops, q_in)
    
    # We need to compute gradients of pos_x, pos_y, pos_z w.r.t q
    # We can do this efficiently by creating a grad_output vectors
    
    jacobians = torch.zeros(batch_size, 3, n_joints, device=q_deg.device)
    
    # Vectorized Jacobian computation
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
    """
    Batch IK solver using PyTorch.
    
    Args:
        urdf_info: from parse_urdf
        targets: (batch_size, 3) tensor or array
        q_init: (batch_size, n_joints) tensor or None
        verbose: if True, prints progress every `log_interval` iterations
        log_interval: iteration interval for verbose logging
    """
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
    else:
        targets = targets.to(device)
        
    batch_size = targets.shape[0]
    n_joints = urdf_info["n_joints"]
    
    chain_ops, limits = urdf_chain_to_torch(urdf_info, device=device)
    
    # Limits are in radians in urdf_info usually, but let's check
    # parse_urdf returns radians for limits
    lows = torch.rad2deg(limits[:, 0])
    highs = torch.rad2deg(limits[:, 1])
    
    # Initialize q (node-style policy)
    if q_init is None:
        # Node: random in small box [-45, 45] then clamp
        q = (torch.rand(batch_size, n_joints, device=device) * 90.0) - 45.0
    else:
        if not isinstance(q_init, torch.Tensor):
            q = torch.tensor(q_init, dtype=torch.float32, device=device)
        else:
            q = q_init.to(device)
        # Node: add Normal(0,5deg) noise to init
        q = q + torch.randn_like(q) * 5.0

    # Clip to limits
    q = torch.max(torch.min(q, highs), lows)
        
    # Heuristic: base yaw (and we will skip joint0 updates)
    if use_base_yaw and n_joints > 0:
        q[:, 0] = torch.rad2deg(torch.atan2(targets[:, 1], targets[:, 0] + 1e-12))
        
    # Per-sample LR adaptation (node-style)
    lr_vec = torch.full((batch_size, 1), float(lr), dtype=torch.float32, device=device)
    prev_err_norm = torch.full((batch_size,), float("inf"), dtype=torch.float32, device=device)
    first_converged_iter = torch.full((batch_size,), -1, dtype=torch.int32, device=device)

    # Main Loop (fixed iterations for dataset consistency)
    for it in range(max_iter):
        pos = forward_kinematics_batch(chain_ops, q)
        err = targets - pos
        err_norm = torch.norm(err, dim=1)
        if verbose and (it % log_interval == 0 or it == max_iter - 1):
            mean_err = err_norm.mean().item()
            max_err = err_norm.max().item()
            print(f"[IK batch] iter {it:04d}/{max_iter} | mean err {mean_err:.6f} | max err {max_err:.6f}")

        # Record first convergence iteration (optional)
        if return_stats:
            newly = (first_converged_iter < 0) & (err_norm <= tol)
            first_converged_iter[newly] = int(it + 1)
        
        # Check convergence (not easy to break early in batch without masking, so we run fixed iters usually,
        # or we just let it run. For dataset gen, running fixed iters is fine/consistent.)
        
        # Jacobian
        J = compute_jacobian_batch(chain_ops, q) # (B, 3, N)
        Jt = J.transpose(1, 2) # (B, N, 3)
        
        # DLS Step: dq = J^T (J J^T + lambda I)^-1 err
        # Compute JJ^T
        JJt = torch.matmul(J, Jt) # (B, 3, 3)
        
        # Add damping
        eye = torch.eye(3, device=device).unsqueeze(0)
        damped = JJt + lam * eye
        
        # Solve linear system (damped) * x = err -> find x
        # x = (JJt + lam I)^-1 err
        # Using torch.linalg.solve is better than inv
        # err is (B, 3) -> unsqueeze to (B, 3, 1)
        err_unsqueezed = err.unsqueeze(2)
        
        try:
            rhs = torch.linalg.solve(damped, err_unsqueezed) # (B, 3, 1)
        except RuntimeError:
            # Fallback for singular matrices if solve fails (add more damping or just noise)
            damped = damped + 1e-2 * eye
            rhs = torch.linalg.solve(damped, err_unsqueezed)

        dq = torch.matmul(Jt, rhs).squeeze(2) # (B, N, 1) -> (B, N)
        
        # Node-style: do not update joint 0 (base yaw fixed)
        if use_base_yaw and n_joints > 0:
            dq[:, 0] = 0.0
        
        # Node-style LR adaptation: halve LR where diverging
        diverging = err_norm > (prev_err_norm * 1.2)
        if diverging.any():
            lr_vec[diverging.unsqueeze(1)] *= 0.5
        prev_err_norm = err_norm

        q = q + lr_vec * dq
        
        # Clip to limits
        q = torch.max(torch.min(q, highs), lows)
        
        # Re-apply base yaw
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
