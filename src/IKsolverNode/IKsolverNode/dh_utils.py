import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

def robot_dh(joint_types, effector_type):
    config = []
    prev_type = None
    

    for i, joint_type in enumerate(joint_types):
        next_type = joint_types[i + 1] if i < len(joint_types) - 1 else None
        prev_type = joint_types[i - 1] if i > 0 else None
        if i == 0:
            d = 0.1769
            a = 0.0
            alpha = -np.pi/2
        elif joint_type == "rot180" and next_type == "rot180":
            d = 0.0
            a = 0.112
            alpha = 0
        elif joint_type == "rot180" and next_type == "rot360" and prev_type == "rot180":
            d = 0.0
            a = 0.0
            alpha = np.pi/2
        elif joint_type == "rot180" and next_type == "rot360" and prev_type == "rot360":
            d = 0.0
            a = 0.0
            alpha = np.pi/2

        elif joint_type == "rot360" and next_type == "rot180" and i != 0:
            d = 0.23665
            a = 0.0
            alpha = -np.pi/2
        elif joint_type == "rot360" and next_type == "rot360" and i != 0:
            d = 0.19265
            a = 0.0
            alpha = 0
        elif next_type is None:
            if joint_type == "rot360":
                d = 0.19265
                a = 0.0
                alpha = 0
            elif joint_type == "rot180":
                if prev_type == "rot180":
                    d = 0.0
                    a = 0.18
                    alpha = 0.0
                else:
                    d = 0.0
                    a = 0.18
                    alpha = np.pi/2

        config.append({'type': joint_type, 'd': d, 'a': a, 'alpha': alpha})
        prev_type = joint_type

    return config


def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,     ca,    d],
        [0,        0,      0,    1]
    ], dtype=float)


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
