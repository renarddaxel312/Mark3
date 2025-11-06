import numpy as np
import xml.etree.ElementTree as ET
from IKsolverNode.dh_utils import dh_matrix
import os


def is_reachable(config, pos):
    reach = sum(abs(j['a']) + abs(j['d']) for j in config)
    return np.linalg.norm(pos) <= reach


def parse_urdf(urdf_path_or_string):
    if os.path.isfile(urdf_path_or_string):
        tree = ET.parse(urdf_path_or_string)
        root = tree.getroot()
    else:
        root = ET.fromstring(urdf_path_or_string)
    
    all_joints = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        joint_type = joint.get('type')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        
        origin = joint.find('origin')
        if origin is not None:
            xyz = origin.get('xyz', '0 0 0').split()
            rpy = origin.get('rpy', '0 0 0').split()
            transform = {
                'xyz': np.array([float(x) for x in xyz]),
                'rpy': np.array([float(x) for x in rpy])
            }
        else:
            transform = {
                'xyz': np.array([0.0, 0.0, 0.0]),
                'rpy': np.array([0.0, 0.0, 0.0])
            }
        
        limits = None
        if joint_type == 'revolute':
            limit = joint.find('limit')
            if limit is not None:
                limits = (float(limit.get('lower')), float(limit.get('upper')))
            else:
                limits = (-np.pi, np.pi)
        
        all_joints[name] = {
            'type': joint_type,
            'parent': parent,
            'child': child,
            'transform': transform,
            'limits': limits
        }

    chain = []
    joint_names = []
    joint_limits = []
    
    current_link = 'base_link'
    visited = set()
    
    while True:
        next_joint = None
        for jname, jinfo in all_joints.items():
            if jname not in visited and jinfo['parent'] == current_link:
                next_joint = (jname, jinfo)
                break
        
        if next_joint is None:
            break
        
        jname, jinfo = next_joint
        visited.add(jname)
        
        chain_entry = {
            'name': jname,
            'type': jinfo['type'],
            'transform': jinfo['transform']
        }
        
        if jinfo['type'] == 'revolute':
            chain_entry['joint_index'] = len(joint_names)
            joint_names.append(jname)
            joint_limits.append(jinfo['limits'])
        else:
            chain_entry['joint_index'] = None
        
        chain.append(chain_entry)
        current_link = jinfo['child']
    
    return {
        'joint_names': joint_names,
        'joint_limits': joint_limits,
        'chain': chain,
        'n_joints': len(joint_names)
    }


def xyz_rpy_to_matrix(xyz, rpy):
    x, y, z = xyz
    roll, pitch, yaw = rpy
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    
    return T


def matrix_to_rpy(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.array([roll, pitch, yaw])


def forward_kinematics_urdf(urdf_info, q, return_points=False, return_full_pose=False):
    T = np.eye(4)
    points = [T[:3, 3].copy()]
    
    n_joints = urdf_info['n_joints']
    q_rad = np.deg2rad(q[:n_joints])
    for chain_entry in urdf_info['chain']:
        T_fixed = xyz_rpy_to_matrix(
            chain_entry['transform']['xyz'],
            chain_entry['transform']['rpy']
        )
        T = T @ T_fixed
        
        if chain_entry['type'] == 'revolute':
            joint_idx = chain_entry['joint_index']
            theta = q_rad[joint_idx]
            
            R_joint = np.eye(4)
            R_joint[:3, :3] = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            T = T @ R_joint
            
            if return_points:
                points.append(T[:3, 3].copy())
    
    if return_full_pose:
        return T
    elif return_points:
        points.append(T[:3, 3].copy())
        return np.array(points)
    else:
        return T[:3, 3]


def inverse_kinematics_urdf(urdf_path_or_string, target_pos, target_rpy=None, q_init=None, 
                           max_iter=1000, lr=0.5, tol=1e-4, n_restarts=2, 
                           orientation_weight=0.1):
    urdf_info = parse_urdf(urdf_path_or_string)
    n = urdf_info['n_joints']
    target_pos = np.array(target_pos)
    
    use_orientation = target_rpy is not None
    if use_orientation:
        target_rpy = np.array(target_rpy)
        target_R = xyz_rpy_to_matrix(np.zeros(3), target_rpy)[:3, :3]
    
    best_q, best_err = None, np.inf
    
    def clamp_angles(q):
        q_clamped = q.copy()
        q_rad = np.deg2rad(q)
        
        for i in range(n):
            lower, upper = urdf_info['joint_limits'][i]
            q_rad_clamped = np.clip(q_rad[i], lower, upper)
            q_clamped[i] = np.rad2deg(q_rad_clamped)
        
        return q_clamped
    
    q0_fixed = np.rad2deg(np.arctan2(target_pos[1], target_pos[0]))
    
    for restart in range(n_restarts):

        if q_init is None:
            q = np.random.uniform(-45, 45, size=n)
            q[0] = q0_fixed
        else:
            q = np.array(q_init) + np.random.normal(0, 5, n)
            q[0] = q0_fixed
        
        q = clamp_angles(q)
        prev_err = np.inf
        current_lr = lr
        
        for iteration in range(max_iter):
            if use_orientation:
                T = forward_kinematics_urdf(urdf_info, q, return_full_pose=True)
                pos = T[:3, 3]
                R = T[:3, :3]
            else:
                pos = forward_kinematics_urdf(urdf_info, q)
            
            if np.any(np.isnan(pos)):
                break
            
            error_pos = target_pos - pos
            err_pos_norm = np.linalg.norm(error_pos)
            
            if use_orientation:
                current_rpy = matrix_to_rpy(R)
                error_orient = target_rpy - current_rpy
                error_orient = np.arctan2(np.sin(error_orient), np.cos(error_orient))
                err_orient_norm = np.linalg.norm(error_orient)
                
                err_norm = err_pos_norm + orientation_weight * err_orient_norm
                error_full = np.concatenate([error_pos, orientation_weight * error_orient])
            else:
                err_norm = err_pos_norm
                error_full = error_pos
            
            pos_converged = err_pos_norm < tol
            orient_converged = (not use_orientation) or (err_orient_norm < 0.01)
            
            if pos_converged and orient_converged:
                if restart == 0:
                    if use_orientation:
                        print(f"IK convergée en {iteration} itérations (pos: {err_pos_norm:.6f}m, orient: {err_orient_norm:.4f}rad)")
                    else:
                        print(f"IK convergée en {iteration} itérations (err: {err_norm:.6f}m)")
                return clamp_angles(q)
            
            if use_orientation:
                J = np.zeros((6, n))
            else:
                J = np.zeros((3, n))
            
            eps = 1e-4
            
            for i in range(n):
                if i == 0:
                    continue
                    
                dq = np.zeros(n)
                dq[i] = eps
                
                if use_orientation:
                    T_plus = forward_kinematics_urdf(urdf_info, q + dq, return_full_pose=True)
                    pos_plus = T_plus[:3, 3]
                    R_plus = T_plus[:3, :3]
                    
                    if not np.any(np.isnan(pos_plus)):
                        J[:3, i] = (pos_plus - pos) / eps
                        rpy_plus = matrix_to_rpy(R_plus)
                        J[3:, i] = (rpy_plus - current_rpy) / eps
                else:
                    pos_plus = forward_kinematics_urdf(urdf_info, q + dq)
                    
                    if not np.any(np.isnan(pos_plus)):
                        J[:, i] = (pos_plus - pos) / eps
            
            lam = 1e-3
            try:
                if use_orientation:
                    J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
                else:
                    J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3))
                
                dq = current_lr * (J_pinv @ error_full)
                q += np.rad2deg(dq)
                q[0] = q0_fixed
                q = clamp_angles(q)
            except np.linalg.LinAlgError:
                break
            
            if err_norm > prev_err * 1.2:
                current_lr *= 0.5
            
            prev_err = err_norm
        
        if err_norm < best_err:
            best_q = q.copy()
            best_err = err_norm
    
    if best_err >= tol:
        if use_orientation:
            print(f"IK URDF non convergée après {max_iter} itérations (meilleur err: {best_err:.4f})")
        else:
            print(f"IK URDF non convergée après {max_iter} itérations (meilleur err: {best_err:.4f}m)")
    else:
        if use_orientation:
            print(f"IK URDF convergée (err: {best_err:.6f})")
        else:
            print(f"IK URDF convergée (err: {best_err:.6f}m)")
    
    return clamp_angles(best_q if best_q is not None else np.zeros(n))
