import numpy as np
import os
import xml.etree.ElementTree as ET


def urdf(config, name="modular_robot"):
    urdf = ['<?xml version="1.0" ?>', f'<robot name="{name}">']

    urdf.append('  <link name="base_link">')
    urdf.append('    <visual><geometry><mesh filename="package://IKsolverNode/urdf/meshes/Base.stl" scale="0.001 0.001 0.001"/></geometry></visual>')
    urdf.append('    <collision><geometry><mesh filename="package://IKsolverNode/urdf/meshes/Base.stl" scale="0.001 0.001 0.001"/></geometry></collision>')
    urdf.append('  </link>')

    first_type = config[0]
    first_child = "link_0_bottom"
    urdf.append('  <joint name="base_to_first" type="fixed">')
    urdf.append('    <parent link="base_link"/>')
    urdf.append(f'    <child link="{first_child}"/>')
    urdf.append('    <origin xyz="0 0 0.03" rpy="0 0 0"/>')
    urdf.append('  </joint>')

    for i, joint_type in enumerate(config[:-1]):
        bottom_link = f"link_{i}_bottom"
        top_link = f"link_{i}_top"
        joint_name = f"joint_{i}"
        
        if joint_type == "rot180":
            bottom_mesh = "ElbowBottom.stl"
            top_mesh = "ElbowTop.stl"
            axis = "0 0 1"
            lower, upper = "-1.5708", "1.5708"
        elif joint_type == "rot360":
            bottom_mesh = "WristBottom.stl"
            top_mesh = "WristTop.stl"
            axis = "0 0 1"
            lower, upper = "-3.1416", "3.1416"
        else:
            bottom_mesh, top_mesh = None, None
            axis, lower, upper = "0 0 1", "-1.5708", "1.5708"

        for link_name, mesh in [(bottom_link, bottom_mesh), (top_link, top_mesh)]:
            urdf.append(f'  <link name="{link_name}">')
            if mesh:
                urdf.append(f'    <visual><geometry><mesh filename="package://IKsolverNode/urdf/meshes/{mesh}" scale="0.001 0.001 0.001"/></geometry></visual>')
                urdf.append(f'    <collision><geometry><mesh filename="package://IKsolverNode/urdf/meshes/{mesh}" scale="0.001 0.001 0.001"/></geometry></collision>')
            urdf.append('  </link>')

        urdf.append(f'  <joint name="{joint_name}" type="revolute">')
        urdf.append(f'    <parent link="{bottom_link}"/>')
        urdf.append(f'    <child link="{top_link}"/>')
        urdf.append(f'    <origin xyz="0 0 0" rpy="0 0 0"/>')
        urdf.append(f'    <axis xyz="{axis}"/>')
        urdf.append(f'    <limit lower="{lower}" upper="{upper}" effort="5.0" velocity="1.0"/>')
        urdf.append('  </joint>')
        last_type = None

        if i < len(config) - 1:
            next_bottom = f"link_{i+1}_bottom"
            next_type = config[i + 1]
            prev_type = config[i-1]

            if next_type == "rot180" and joint_type == "rot360":
                urdf.append(f'  <joint name="link_{i}_to_{i+1}" type="fixed">')
                urdf.append(f'    <parent link="{top_link}"/>')
                urdf.append(f'    <child link="{next_bottom}"/>')
                urdf.append(f'    <origin xyz="0 -0.0265 0.1469" rpy="1.57 -1.57 0"/>')
                urdf.append('  </joint>')

            elif next_type == "rot180" and joint_type == "rot180":
                urdf.append(f'  <joint name="link_{i}_to_{i+1}" type="fixed">')
                urdf.append(f'    <parent link="{top_link}"/>')
                urdf.append(f'    <child link="{next_bottom}"/>')
                urdf.append(f'    <origin xyz="0.1125 0 0" rpy="0 0 0"/>')
                urdf.append('  </joint>')

            elif next_type == "rot360" and joint_type == "rot180":
                urdf.append(f'  <joint name="link_{i}_to_{i+1}" type="fixed">')
                urdf.append(f'    <parent link="{top_link}"/>')
                urdf.append(f'    <child link="{next_bottom}"/>')
                urdf.append(f'    <origin xyz="0.0895 0 -0.0258" rpy="-1.57 0 -1.57"/>')
                urdf.append('  </joint>')
            
            elif next_type == "rot360" and joint_type == "rot360":
                urdf.append(f'  <joint name="link_{i}_to_{i+1}" type="fixed">')
                urdf.append(f'    <parent link="{top_link}"/>')
                urdf.append(f'    <child link="{next_bottom}"/>')
                urdf.append(f'    <origin xyz="0 0 0.1236" rpy="0 0 0"/>')
                urdf.append('  </joint>')
            
            prev_joint = joint_type

    penultimate_joint = config[-3]
    ultimate_joint = config[-2]
    last_top = f"link_{len(config)-2}_top"
    if ultimate_joint == "rot360":
        urdf.append(f'  <link name="tool_link">')
        urdf.append(f'    <visual><geometry><mesh filename="package://IKsolverNode/urdf/meshes/Gripper.stl" scale="0.001 0.001 0.001"/></geometry></visual>')
        urdf.append(f'    <collision><geometry><mesh filename="package://IKsolverNode/urdf/meshes/Gripper.stl" scale="0.001 0.001 0.001"/></geometry></collision>')
        urdf.append(f'  </link>')
        urdf.append(f'  <joint name="end_effector" type="fixed">')
        urdf.append(f'    <parent link="{last_top}"/>')
        urdf.append(f'    <child link="tool_link"/>')
        urdf.append(f'    <origin xyz="-0.0003 -0.0006 0.1914" rpy="0 0 1.57"/>')
        urdf.append(f'  </joint>')
    else:
        urdf.append(f'  <link name="tool_link">')
        urdf.append(f'    <visual><geometry><mesh filename="package://IKsolverNode/urdf/meshes/Gripper.stl" scale="0.001 0.001 0.001"/></geometry></visual>')
        urdf.append(f'    <collision><geometry><mesh filename="package://IKsolverNode/urdf/meshes/Gripper.stl" scale="0.001 0.001 0.001"/></geometry></collision>')
        urdf.append(f'  </link>')
        urdf.append(f'  <joint name="end_effector" type="fixed">')
        urdf.append(f'    <parent link="{last_top}"/>')
        urdf.append(f'    <child link="tool_link"/>')
        urdf.append(f'    <origin xyz="0.157 -0.0004 -0.0252" rpy="0 1.57 0"/>')
        urdf.append(f'  </joint>')

    urdf.append('</robot>')
    return "\n".join(urdf)


def compute_reachability_sphere(config):

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