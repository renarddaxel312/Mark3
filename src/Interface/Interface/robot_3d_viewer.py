#!/usr/bin/env python3
import numpy as np
import xml.etree.ElementTree as ET
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
import vtk

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as _QVTKRenderWindowInteractor
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as _QVTKRenderWindowInteractor


class QVTKRenderWindowInteractor(_QVTKRenderWindowInteractor):
    
    def mousePressEvent(self, ev):
        try:
            super().mousePressEvent(ev)
        except AttributeError:
            ctrl, shift = self._GetCtrlShift(ev)
            repeat = 0
            
            if ev.button() == Qt.MouseButton.LeftButton:
                self._Iren.SetEventInformationFlipY(ev.x(), ev.y(), ctrl, shift, chr(0), repeat, None)
                self._Iren.LeftButtonPressEvent()
                self._ActiveButton = Qt.MouseButton.LeftButton
            elif ev.button() == Qt.MouseButton.RightButton:
                self._Iren.SetEventInformationFlipY(ev.x(), ev.y(), ctrl, shift, chr(0), repeat, None)
                self._Iren.RightButtonPressEvent()
                self._ActiveButton = Qt.MouseButton.RightButton
            elif ev.button() == Qt.MouseButton.MiddleButton:
                self._Iren.SetEventInformationFlipY(ev.x(), ev.y(), ctrl, shift, chr(0), repeat, None)
                self._Iren.MiddleButtonPressEvent()
                self._ActiveButton = Qt.MouseButton.MiddleButton
    
    def mouseReleaseEvent(self, ev):
        try:
            super().mouseReleaseEvent(ev)
        except AttributeError:
            ctrl, shift = self._GetCtrlShift(ev)
            
            if ev.button() == Qt.MouseButton.LeftButton:
                self._Iren.SetEventInformationFlipY(ev.x(), ev.y(), ctrl, shift, chr(0), 0, None)
                self._Iren.LeftButtonReleaseEvent()
            elif ev.button() == Qt.MouseButton.RightButton:
                self._Iren.SetEventInformationFlipY(ev.x(), ev.y(), ctrl, shift, chr(0), 0, None)
                self._Iren.RightButtonReleaseEvent()
            elif ev.button() == Qt.MouseButton.MiddleButton:
                self._Iren.SetEventInformationFlipY(ev.x(), ev.y(), ctrl, shift, chr(0), 0, None)
                self._Iren.MiddleButtonReleaseEvent()
            
            self._ActiveButton = Qt.MouseButton.NoButton


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


def parse_urdf_with_meshes(urdf_path):
    if not os.path.exists(urdf_path):
        print(f"URDF non trouvé: {urdf_path}")
        return None
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    urdf_dir = os.path.dirname(urdf_path)
    
    links_meshes = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        visual = link.find('visual')
        if visual is not None:
            visual_origin = visual.find('origin')
            if visual_origin is not None:
                visual_xyz = [float(x) for x in visual_origin.get('xyz', '0 0 0').split()]
                visual_rpy = [float(x) for x in visual_origin.get('rpy', '0 0 0').split()]
                visual_transform = {'xyz': np.array(visual_xyz), 'rpy': np.array(visual_rpy)}
            else:
                visual_transform = {'xyz': np.array([0.0, 0.0, 0.0]), 'rpy': np.array([0.0, 0.0, 0.0])}
            
            geometry = visual.find('geometry')
            if geometry is not None:
                mesh_elem = geometry.find('mesh')
                if mesh_elem is not None:
                    mesh_filename = mesh_elem.get('filename')
                    if mesh_filename.startswith('package://'):
                        mesh_filename = mesh_filename.replace('package://IKsolverNode/', '')
                        package_dir = os.path.dirname(urdf_dir)
                        mesh_path = os.path.join(package_dir, mesh_filename)
                    else:
                        mesh_path = mesh_filename
                    
                    scale = mesh_elem.get('scale', '1 1 1')
                    scale_vals = [float(x) for x in scale.split()]
                    
                    links_meshes[link_name] = {
                        'mesh_path': mesh_path,
                        'scale': scale_vals,
                        'visual_transform': visual_transform
                    }
    
    all_joints = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        joint_type = joint.get('type')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        
        origin = joint.find('origin')
        if origin is not None:
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
            transform = {'xyz': np.array(xyz), 'rpy': np.array(rpy)}
        else:
            transform = {'xyz': np.array([0.0, 0.0, 0.0]), 'rpy': np.array([0.0, 0.0, 0.0])}
        
        limits = None
        if joint_type == 'revolute':
            limit = joint.find('limit')
            if limit is not None:
                limits = (float(limit.get('lower')), float(limit.get('upper')))
        
        all_joints[name] = {
            'type': joint_type,
            'parent': parent,
            'child': child,
            'transform': transform,
            'limits': limits
        }
    
    chain = []
    joint_names = []
    
    current_link = 'base_link'
    visited = set()
    
    if 'base_link' in links_meshes:
        chain.append({
            'link_name': 'base_link',
            'mesh_info': links_meshes['base_link'],
            'joint_index': None,
            'type': 'fixed',
            'transform': {'xyz': np.array([0.0, 0.0, 0.0]), 'rpy': np.array([0.0, 0.0, 0.0])}
        })
    
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
        
        child_link = jinfo['child']
        mesh_info = links_meshes.get(child_link, None)
        
        chain_entry = {
            'link_name': child_link,
            'joint_name': jname,
            'type': jinfo['type'],
            'transform': jinfo['transform'],
            'mesh_info': mesh_info
        }
        
        if jinfo['type'] == 'revolute':
            chain_entry['joint_index'] = len(joint_names)
            joint_names.append(jname)
        else:
            chain_entry['joint_index'] = None
        
        chain.append(chain_entry)
        current_link = jinfo['child']
    
    return {'joint_names': joint_names, 'chain': chain, 'n_joints': len(joint_names)}


class Robot3DViewer(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.118, 0.118, 0.118)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        self.setup_lights()
        
        self.setup_axes()
        
        self.setup_grid()
        
        self.text_actor = self.setup_text_display()
        
        self.urdf_info = None
        self.urdf_path = None
        self.actors = []
        self.current_q = None
        self.end_effector_pos = np.array([0.0, 0.0, 0.0])
        
        self.interactor.Initialize()
    
    def setup_lights(self):
        light1 = vtk.vtkLight()
        light1.SetPosition(1, 1, 1)
        light1.SetFocalPoint(0, 0, 0)
        light1.SetColor(1.0, 1.0, 1.0)
        light1.SetIntensity(0.8)
        self.renderer.AddLight(light1)
        
        light2 = vtk.vtkLight()
        light2.SetPosition(-1, -1, 0.5)
        light2.SetFocalPoint(0, 0, 0)
        light2.SetColor(0.7, 0.7, 0.9)
        light2.SetIntensity(0.4)
        self.renderer.AddLight(light2)
    
    def setup_axes(self):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.1, 0.1, 0.1)
        axes.SetShaftTypeToCylinder()
        axes.SetNormalizedShaftLength(0.8, 0.8, 0.8)
        axes.SetNormalizedTipLength(0.2, 0.2, 0.2)
        
        axes.AxisLabelsOff()
        
        self.renderer.AddActor(axes)
    
    def setup_grid(self):
        grid_size = 2.0
        grid_spacing = 0.1
        
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        n_lines = int(grid_size / grid_spacing) + 1
        start = -grid_size / 2
        
        for i in range(n_lines):
            y = start + i * grid_spacing
            id1 = points.InsertNextPoint(start, y, 0)
            id2 = points.InsertNextPoint(-start, y, 0)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id1)
            line.GetPointIds().SetId(1, id2)
            lines.InsertNextCell(line)
        
        for i in range(n_lines):
            x = start + i * grid_spacing
            id1 = points.InsertNextPoint(x, start, 0)
            id2 = points.InsertNextPoint(x, -start, 0)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id1)
            line.GetPointIds().SetId(1, id2)
            lines.InsertNextCell(line)
        
        grid = vtk.vtkPolyData()
        grid.SetPoints(points)
        grid.SetLines(lines)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(grid)
        
        grid_actor = vtk.vtkActor()
        grid_actor.SetMapper(mapper)
        grid_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        grid_actor.GetProperty().SetLineWidth(1)
        grid_actor.GetProperty().SetOpacity(0.3)
        
        self.renderer.AddActor(grid_actor)
    
    def setup_text_display(self):
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput("Position de l'effecteur: x = 0.000 y = 0.000 z = 0.000")
        
        text_actor.SetPosition(10, 10)
        text_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        
        text_prop = text_actor.GetTextProperty()
        text_prop.SetFontSize(16)
        text_prop.SetColor(1.0, 1.0, 1.0)
        text_prop.SetFontFamilyToArial()
        text_prop.SetBold(True)
        text_prop.SetShadow(True)
        
        self.renderer.AddActor2D(text_actor)
        return text_actor
    
    def update_text_display(self):
        x, y, z = self.end_effector_pos
        text = f"Position de l'effecteur: x = {x:.3f} y = {y:.3f} z = {z:.3f}"
        self.text_actor.SetInput(text)
    
    def set_urdf_path(self, urdf_path):
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors = []
        
        self.vtk_widget.GetRenderWindow().Render()
        
        self.urdf_path = urdf_path
        self.urdf_info = parse_urdf_with_meshes(urdf_path)
        
        if self.urdf_info:
            self.current_q = [0.0] * self.urdf_info['n_joints']
            self.end_effector_pos = np.array([0.0, 0.0, 0.0])
    
    def update_view(self, q_deg):
        if self.urdf_info is None:
            return
        
        try:
            for actor in self.actors:
                self.renderer.RemoveActor(actor)
        except Exception as e:
            print(f"Erreur lors du nettoyage des actors: {e}")
        self.actors = []
        
        if len(q_deg) < self.urdf_info['n_joints']:
            q_deg = list(q_deg) + [0.0] * (self.urdf_info['n_joints'] - len(q_deg))
        
        q_rad = np.deg2rad(q_deg[:self.urdf_info['n_joints']])
        
        T = np.eye(4)
        
        for chain_entry in self.urdf_info['chain']:
            T_fixed = xyz_rpy_to_matrix(
                chain_entry['transform']['xyz'],
                chain_entry['transform']['rpy']
            )
            T = T @ T_fixed
            
            if chain_entry['type'] == 'revolute' and chain_entry['joint_index'] is not None:
                joint_idx = chain_entry['joint_index']
                theta = q_rad[joint_idx]
                
                R_joint = np.eye(4)
                R_joint[:3, :3] = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                T = T @ R_joint
            
            mesh_info = chain_entry.get('mesh_info')
            if mesh_info:
                if os.path.exists(mesh_info['mesh_path']):
                    actor = self.create_mesh_actor(mesh_info, T)
                    if actor:
                        self.actors.append(actor)
                        self.renderer.AddActor(actor)
                    else:
                        print(f"Échec création actor pour: {mesh_info['mesh_path']}")
                else:
                    print(f"Mesh non trouvé: {mesh_info['mesh_path']}")
        
        self.end_effector_pos = T[:3, 3]
        self.update_text_display()
        
        if len(self.actors) > 0 and self.current_q is None:
            self.renderer.ResetCamera()
            camera = self.renderer.GetActiveCamera()
            camera.Elevation(20)
            camera.Azimuth(45)
            camera.Zoom(1.2)
        
        self.current_q = q_deg
        self.vtk_widget.GetRenderWindow().Render()
    
    def create_mesh_actor(self, mesh_info, transform):
        try:
            if not os.path.exists(mesh_info['mesh_path']):
                print(f"Fichier mesh introuvable: {mesh_info['mesh_path']}")
                return None
            
            reader = vtk.vtkSTLReader()
            reader.SetFileName(mesh_info['mesh_path'])
            reader.Update()
            
            if reader.GetOutput().GetNumberOfPoints() == 0:
                print(f"Mesh vide: {mesh_info['mesh_path']}")
                return None
            
            scale = mesh_info['scale']
            T_scale = np.eye(4)
            T_scale[0, 0] = scale[0]
            T_scale[1, 1] = scale[1]
            T_scale[2, 2] = scale[2]
            
            if 'visual_transform' in mesh_info:
                T_visual = xyz_rpy_to_matrix(
                    mesh_info['visual_transform']['xyz'],
                    mesh_info['visual_transform']['rpy']
                )
            else:
                T_visual = np.eye(4)
            
            T_final = transform @ T_visual @ T_scale
            
            vtk_transform = vtk.vtkTransform()
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vtk_matrix.SetElement(i, j, T_final[i, j])
            vtk_transform.SetMatrix(vtk_matrix)
            
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(reader.GetOutputPort())
            transform_filter.SetTransform(vtk_transform)
            transform_filter.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transform_filter.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            actor.GetProperty().SetColor(0.8, 0.8, 0.8)
            actor.GetProperty().SetSpecular(0.3)
            actor.GetProperty().SetSpecularPower(20)
            actor.GetProperty().SetAmbient(0.3)
            actor.GetProperty().SetDiffuse(0.7)
            
            return actor
        except Exception as e:
            print(f"Erreur lors du chargement du mesh {mesh_info['mesh_path']}: {e}")
            return None
    
    def update_joints(self, q_deg):
        self.update_view(q_deg)
