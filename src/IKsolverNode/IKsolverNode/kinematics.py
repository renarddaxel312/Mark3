import numpy as np
import xml.etree.ElementTree as ET
from IKsolverNode.dh_utils import dh_matrix
import os


def compute_reachability_sphere(config):
    """
    Calcule la sphère de reachability du robot.
    Retourne (center, radius) où center est le centre de la sphère et radius son rayon.
    """
    # Calculer la portée maximale (somme des longueurs des segments)
    max_reach = sum(abs(j.get('a', 0)) + abs(j.get('d', 0)) for j in config)
    
    # Le centre est généralement à l'origine (base du robot)
    # Mais on peut le décaler si nécessaire selon la configuration
    center = np.array([0.0, 0.0, 0.0])
    
    # Ajuster le centre si le premier segment a un offset
    if config and 'd' in config[0]:
        center[2] = config[0].get('d', 0.0) * 0.5  # Décalage partiel selon d1
    
    return center, max_reach


def get_default_exclusion_zones(config):
    """
    Retourne les zones d'exclusion par défaut pour éviter les auto-collisions.
    Chaque zone est une sphère définie par {'center': [x, y, z], 'radius': r}
    """
    exclusion_zones = []
    
    # Zone 1: Base du robot (sphère autour de l'origine)
    base_radius = 0.15  # Rayon de sécurité autour de la base (en mètres)
    exclusion_zones.append({
        'center': np.array([0.0, 0.0, 0.0]),
        'radius': base_radius,
        'name': 'base'
    })
    
    # Zone 2: Espace sous la base (Z négatif)
    # Permettre d'aller jusqu'à la hauteur de la base (-0.106 m)
    # Cette zone bloque seulement vraiment en dessous de la base
    exclusion_zones.append({
        'center': np.array([0.0, 0.0, -0.15]),  # Plus bas que la base
        'radius': 0.1,  # Rayon plus petit pour ne bloquer que vraiment en dessous
        'name': 'below_base'
    })
    
    # Zone 3: Zone de coude (si le robot a des segments)
    if len(config) >= 3:
        # Estimer la position du coude depuis la config
        elbow_z = sum(config[i].get('d', 0.0) for i in range(min(2, len(config))))
        elbow_radius = 0.1  # Rayon autour du coude
        exclusion_zones.append({
            'center': np.array([0.0, 0.0, elbow_z]),
            'radius': elbow_radius,
            'name': 'elbow_zone'
        })
    
    return exclusion_zones


def is_reachable(config, pos, exclusion_zones=None, sphere_center=None, sphere_radius=None):
    """
    Vérifie si une position (x, y, z) est accessible par le robot.
    
    Args:
        config: Configuration du robot (liste de dict avec 'a' et 'd')
        pos: Position cible [x, y, z] en mètres
        exclusion_zones: Liste optionnelle de zones d'exclusion. 
                        Si None, utilise les zones par défaut.
                        Chaque zone est {'center': [x, y, z], 'radius': r}
        sphere_center: Centre de la sphère de reachability (optionnel, calculé si None)
        sphere_radius: Rayon de la sphère de reachability (optionnel, calculé si None)
    
    Returns:
        bool: True si la position est accessible, False sinon
    """
    pos = np.array(pos)
    
    # 1. Calculer la sphère de reachability si non fournie
    if sphere_center is None or sphere_radius is None:
        sphere_center, sphere_radius = compute_reachability_sphere(config)
    
    # 2. Vérifier si le point est dans la sphère principale
    dist_from_center = np.linalg.norm(pos - sphere_center)
    if dist_from_center > sphere_radius:
        return False
    
    # 3. Vérifier les zones d'exclusion (auto-collision)
    if exclusion_zones is None:
        exclusion_zones = get_default_exclusion_zones(config)
    
    for zone in exclusion_zones:
        zone_center = np.array(zone['center'])
        zone_radius = zone['radius']
        dist_to_zone = np.linalg.norm(pos - zone_center)
        
        if dist_to_zone < zone_radius:
            return False  # Point dans une zone d'exclusion
    
    return True


def compute_reachability_intervals(config, exclusion_zones=None, sphere_center=None, sphere_radius=None):
    """
    Calcule les intervalles min/max pour X, Y, Z en fonction de la reachability.
    
    Args:
        config: Configuration du robot
        exclusion_zones: Zones d'exclusion (optionnel)
        sphere_center: Centre de la sphère (optionnel)
        sphere_radius: Rayon de la sphère (optionnel)
    
    Returns:
        dict: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
    """
    if sphere_center is None or sphere_radius is None:
        sphere_center, sphere_radius = compute_reachability_sphere(config)
    
    if exclusion_zones is None:
        exclusion_zones = get_default_exclusion_zones(config)
    
    # Initialiser avec les limites de la sphère principale
    x_min = sphere_center[0] - sphere_radius
    x_max = sphere_center[0] + sphere_radius
    y_min = sphere_center[1] - sphere_radius
    y_max = sphere_center[1] + sphere_radius
    z_min = sphere_center[2] - sphere_radius
    z_max = sphere_center[2] + sphere_radius
    
    # Ajuster selon les zones d'exclusion
    # Pour chaque zone, on exclut l'espace qu'elle occupe
    for zone in exclusion_zones:
        zone_center = np.array(zone['center'])
        zone_radius = zone['radius']
        
        # Si la zone est proche des limites, ajuster les intervalles
        # Zone de base : ne pas bloquer complètement, juste éviter les collisions directes
        if zone['name'] == 'base':
            # La zone de base n'empêche pas d'aller légèrement en dessous
            pass
        elif zone['name'] == 'below_base':
            # Bloquer seulement vraiment en dessous de la base
            z_min = max(z_min, zone_center[2] + zone_radius)
    
    # Fixer z_min à la hauteur de la base (-0.106 m)
    base_height = -0.106
    z_min = base_height
    
    return {
        'x': (float(x_min), float(x_max)),
        'y': (float(y_min), float(y_max)),
        'z': (float(z_min), float(z_max))
    }


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
