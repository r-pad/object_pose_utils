# -*- coding: utf-8 -*-
"""
Created on a Thursday long long ago

@author: bokorn
"""

import cv2
import numpy as np
import copy
from quat_math import quaternion_matrix, rotation_matrix
from object_pose_utils.datasets.image_processing import get_bbox_label

def rotateImage(img, theta):
    if(len(img.shape) == 2 or img.shape[2] == 1):
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR
            
    (oldY,oldX) = img.shape[:2]
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=np.rad2deg(theta), scale=1.0)

    newX,newY = oldX,oldY
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    newX,newY = (abs(sin_theta*newY) + abs(cos_theta*newX),abs(sin_theta*newX) + abs(cos_theta*newY))

    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx
    M[1,2] += ty

    img_rot = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)), flags=interp)
    return img_rot

def rotateMetaData(meta_data, theta):
    meta_data_rot = copy.deepcopy(meta_data)
    (oldY,oldX) = meta_data['mask'].shape[:2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    newX,newY = (abs(sin_theta*oldY) + abs(cos_theta*oldX),abs(sin_theta*oldX) + abs(cos_theta*oldY))
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    cx = meta_data_rot['camera_cx']
    cy = meta_data_rot['camera_cy']
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=np.rad2deg(theta), scale=1.0)
    center = M.dot(np.array([cx, cy, 1]).T)

    meta_data_rot['camera_cx'] = center[0] + tx
    meta_data_rot['camera_cy'] = center[1] + ty
    
    meta_data_rot['transform_mat'] = rotation_matrix(theta, [0,0,-1]).dot(meta_data['transform_mat'])
    
    meta_data_rot['mask'] = rotateImage(meta_data['mask'].astype(np.uint8), theta) > .5
    meta_data_rot['bbox'] = get_bbox_label(meta_data_rot['mask'])
    return meta_data_rot

def rotateQuaternion(quat, theta):
    quat_rot = quat_math.quaternion_multiply(quat_math.quaternion_about_axis(theta, [0,0,-1]), quat)
    return quat_rot

def rotatePoints(points, theta, meta_data):
    mat_rot = torch.Tensor(rotation_matrix(theta, [0,0,-1])[:3,:3])
    points_rot = torch.mm(mat_rot, points.t()).t()
    return points_rot

def InplaneRotator(meta_data, img, depth, points, 
                   theta = None):
    if(theta is None):
        theta = np.random.rand()*2.*np.pi

    meta_data = rotateMetaData(meta_data, theta)
    if(img is not None):
        img = rotateImage(img, theta)
    if(depth is not None):
        depth = rotateImage(depth, theta)

    return meta_data, img, depth, points 
