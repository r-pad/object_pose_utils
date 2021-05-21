'''Setup renderer'''

from quat_math import quaternion_matrix
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import pyrender

import numpy as np

def createScene(obj_filename, meta, scale = True):
    scene = pyrender.Scene(ambient_light=np.ones(3))

    obj_trimesh = trimesh.load(obj_filename)
    if scale:
        obj_trimesh.units = 'mm'
        obj_trimesh.convert_units('m')
    obj_mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
    obj_pose = np.eye(4)
    obj_node = scene.add(obj_mesh, pose=obj_pose)

    camera = pyrender.IntrinsicsCamera(fx=meta['camera_fx'],
                                       fy=meta['camera_fy'],
                                       cx=meta['camera_cx'],
                                       cy=meta['camera_cy'])
    camera_pose = quaternion_matrix([1,0,0,0])
    scene.add(camera, pose=camera_pose)
    return scene, obj_node

class ObjectRenderer(object):
    def __init__(self, mesh_filename, camera_meta, 
                 width = 640, height = 480, point_size = 1.0,
                 scale_mm_mesh = True):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=width, viewport_height=height, 
            point_size=point_size)
        
        self.scene, self.obj_node = createScene(mesh_filename, 
            camera_meta, scale_mm_mesh)
        
    def render_pose(self, pose_mat):
        self.scene.set_pose(self.obj_node, pose_mat)
        return self.renderer.render(self.scene)
