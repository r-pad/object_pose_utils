# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:38:19 2018

@author: bokorn
"""
import os
import cv2
import numpy as np
import numpy.ma as ma

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from object_pose_utils.utils.image_preprocessing import cropBBox, transparentOverlay
from quat_math import quaternion_from_matrix

from enum import Enum, auto
class OutputTypes(Enum):
    # Image Types
    IMAGE = auto()
    IMAGE_MASKED = auto()
    IMAGE_CROPPED = auto()
    IMAGE_MASKED_CROPPED = auto()
    # Object Types
    OBJECT_LABEL = auto()
    MODEL_POINTS = auto()
    # Segmentaion Types
    BBOX = auto()
    MASK = auto()
    # Ground Truth Types
    TRANSFORM_MATRIX = auto()
    ROTATION_MATRIX = auto()
    QUATERNION = auto()
    TRANSLATION = auto()
    MODEL_POINTS_TRANSFORMED = auto()
    # Depth Types
    DEPTH_IMAGE = auto()
    DEPTH_IMAGE_MASKED = auto()
    DEPTH_IMAGE_CROPPED = auto()
    DEPTH_IMAGE_MASKED_CROPPED = auto()
    DEPTH_POINTS = auto()
    DEPTH_POINTS_MASKED = auto()

IMAGE_OUTPUTS = set(OutputTypes.IMAGE, 
                    OutputTypes.IMAGE_MASKED, 
                    OutputTypes.IMAGE_CROPPED, 
                    OutputTypes.IMAGE_MASKED_CROPPED,
                    )
DEPTH_OUTPUTS = set(OutputTypes.DEPTH_IMAGE,
                    OutputTypes.DEPTH_IMAGE_MASKED, 
                    OutputTypes.DEPTH_IMAGE_MASKED, 
                    OutputTypes.DEPTH_IMAGE_MASKED, 
                    OutputTypes.DEPTH_IMAGE_MASKED, 
                    OutputTypes.DEPTH_IMAGE_MASKED
                    )
TRANSFORM_OUTPUTS = set(OutputTypes.TRANSFORM_MATRIX,
                        OutputTypes.ROTATION_MATRIX,
                        OutputTypes.QUATERNION,
                        OutputTypes.TRANSLATION,
                        )
DEPTH_POINT_OUTPUTS = set(OutputTypes.DEPTH_POINTS,
                          OutputTypes.DEPTH_POINTS_MASKED,
                          )
MODEL_POINT_OUTPUTS = set(OutputTypes.MODEL_POINTS,
                          OutputTypes.MODEL_POINTS_TRANSFORMED,
                          )
MASK_OUTPUTS = set(OutputTypes.MASK, 
                   OutputTypes.IMAGE_MASKED, 
                   OutputTypes.IMAGE_MASKED_CROPPED, 
                   OutputTypes.DEPTH_IMAGE_MASKED, 
                   OutputTypes.DEPTH_IMAGE_MASKED_CROPPED, 
                   OutputTypes.DEPTH_POINTS_MASKED,
                   )

BBOX_OUTPUTS = set(OutputTypes.BBOX, 
                   OutputTypes.IMAGE_CROPPED, 
                   OutputTypes.IMAGE_MASKED_CROPPED, 
                   OutputTypes.DEPTH_IMAGE_CROPPED, 
                   OutputTypes.DEPTH_IMAGE_MASKED_CROPPED,
                   OutputTypes.DEPTH_POINTS_MASKED,
                   )

CAMERA_MATRIX_OUTPUTS = set(OutputTypes.DEPTH_POINTS,
                            OutputTypes.DEPTH_POINTS_MASKED,
                            )

class PoseImageDataset(Dataset):
    def __init__(self, 
                 image_size,
                 output_data = [
                                OutputTypes.IMAGE_CROPPED,
                                OutputTypes.QUATERNION,
                               ]
                 preprocessor = None,
                 *args, **kwargs):
        super(PoseImageDataset, self).__init__()
        self.image_size = image_size
        self.output_types = set(output_data)
        self.preprocessor = preprocessor
        self.REMOVE_MASK = True
        self.backgroud_fill = 255.0
        self.boarder_width = 0.0
        self.IMAGE_CONTAINS_MASK = False
        self.BBOX_FROM_MASK = False
        self.num_points = -1

    def __getitem__(self, index):
        output = []
        need_mask = not self.IMAGE_CONTAINS_MASK and len(output_data & MASK_OUTPUTS) > 0
        need_bbox = not self.BBOX_FROM_MASK and len(output_data & BBOX_OUTPUTS) > 0
        need_camera_matrix = len(output_data & CAMERA_MATRIX_OUTPUTS) >  0
        meta_data = getMetaData(mask = need_mask, bbox = need_bbox, camera_matrix = need_camera_matrix)
        if(need_mask and self.IMAGE_CONTAINS_MASK):
            img = self.getImage(index)
            meta_data['mask'] = img[:,:,3]

        # Could build a output function at init to speed this up if its slow.
        for output_type in self.output_types:
            if(output_type is in IMAGE_OUTPUTS):
                if('img' is not in locals()):
                    img = self.getImage(index)
                outputs.append(self.processImage(img.copy(), meta_data, output_type))
            elif(output_type is in DEPTH_OUTPUTS):
                if('depth' is not in locals()):
                    depth = self.getDepthImage(index)
                outputs.append(self.processDepthImage(depth.copy(), meta_data, output_type))
            elif(output_type is in MODEL_POINT_OUTPUTS):
                if('points' is not in locals()):
                    points = self.getModelPoints(meta_data['object_label'])
                outputs.append(self.processModelPoints(points.copy(), meta_data, output_type))
            elif(output_type is in TRANSFORM_OUTPUTS):
                outputs.append(self.processTransform(meta_data, output_type))
            elif(output_type is OutputTypes.OBJECT_LABEL):
                outputs.append(meta_data['object_label'])
            elif(output_type is OutputTypes.MASK):
                outputs.append(meta_data['mask'])
            elif(output_type is OutputTypes.BBOX):
                outputs.append(meta_data['bbox'])
            else:
                raise ValueError('Invalid Output Type {}'.format(output_type))

        if(self.preprocessor is not None):
            outputs = self.preprocessor(outputs, meta_data, self.output_types)
    
        return tuple(outputs)

    def processImage(self, img, meta_data, output_type):
        if(output_type is in MASK_OUTPUTS):
            img = np.concatenate([img, meta_data['mask']], axis=2)
            if(self.background_fill is not None):
                image = transparentOverlay(img, self.background_fill, remove_mask=self.remove_mask)
        if(output_type is in BBOX_OUTPUT):
            img, _ = cropBBox(img, meta_data['bbox'], self.boarder_width)
        return img

    def processDepthImage(self, depth, meta_data, output_type):
        if(output_type is in MASK_OUTPUTS):
            depth = ma.masked_array(depth, meta_data['mask'] & depth != 0)

        if(output_type is in BBOX_OUTPUTS):
            depth, corner = cropBBox(depth, meta_data['bbox'], self.boarder_width)
        else:
            corner = (0,0)

        if(output_type is in DEPTH_POINT_OUTPUTS):
            x_size, y_size = depth.shape[:2]
            xmap, ymap = np.meshgrid(np.arange(x_size), np.arange(y_size))
            x_map += corner[0]
            y_map += corner[1]
            choose = depth.flatten().nonzero()[0]
            if len(choose) > self.num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            elif len(choose) > 0:
                choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')
            else:
                return None
            
            depth_choose = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_choose = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_choose = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            z = depth_choose / meta_data['camera_scale']
            x = (ymap_choose - meta_data['camera_cx']) * z / meta_data['camera_fx']
            y = (xmap_choose - meta_data['camera_cy']) * z / meta_data['camera_fy']
            cloud = np.concatenate((x, y, z), axis=1)
        
            return cloud
        
        return depth

    def processModelPoints(self, points, meta_data, output_type):
        if(output_type is OutputTypes.MODEL_POINTS_TRANSFORMED):
            points = points
            R = meta_data['transform_mat'][:3,:3]
            t = meta_data['transform_mat'][:3,3]
            points = np.add(np.dot(points, R.T), t)
        return points

    def processTransform(self, meta_data, output_type): 
        if(output_type is OutputTypes.ROTATION_MATRIX):
            return meta_data['transform_mat'][:3,:3]
        if(output_type is OutputTypes.QUATERNION):
            R = np.eye(4)
            R[:3,:3] = meta_data['transform_mat'][:3,:3]
            return quaternion_from_matrix(R)
        if(output_type is OutputTypes.TRANSLATION):
            return meta_data['transform_mat'][:3,3]
        return meta_data['transform_mat']

    def getDepthImage(self, index):
        raise NotImplementedError('getDepthImage must be implemented by child classes')
    def getImage(self, index):
        raise NotImplementedError('getImage must be implemented by child classes')

    ### Should return dictionary containing {transform_mat, object_label}
    # Optionally containing {mask, bbox, camera_scale, camera_cx, camera_cy, camera_fx, camera_fy}
    def getMetaData(self, mask=False, bbox=False, camera_matrix=False)
        raise NotImplementedError('getModelPoints must be implemented by child classes')
    
    def getModelPoints(self, object_label):
        raise NotImplementedError('getModelPoints must be implemented by child classes')
    
    def __len__(self):
        raise NotImplementedError('__len__ must be implemented by child classes')

