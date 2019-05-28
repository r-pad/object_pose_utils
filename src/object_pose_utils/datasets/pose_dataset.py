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

from enum import Enum

class PoseDataError(Exception):
    pass

class OutputTypes(Enum):
    # Image Types
    IMAGE = 1
    IMAGE_MASKED = 2
    IMAGE_CROPPED = 3
    IMAGE_MASKED_CROPPED = 4
    # Object Types
    OBJECT_LABEL = 5
    MODEL_POINTS = 6
    # Segmentaion Types
    BBOX = 7
    MASK = 8
    # Ground Truth Types
    TRANSFORM_MATRIX = 9
    ROTATION_MATRIX = 10
    QUATERNION = 11
    TRANSLATION = 12
    MODEL_POINTS_TRANSFORMED = 13
    # Depth Types
    DEPTH_IMAGE = 14
    DEPTH_IMAGE_MASKED = 15
    DEPTH_IMAGE_CROPPED = 16
    DEPTH_IMAGE_MASKED_CROPPED = 17
    DEPTH_POINTS = 18
    DEPTH_POINTS_MASKED = 19
    DEPTH_POINTS_AND_INDEXES = 20
    DEPTH_POINTS_MASKED_AND_INDEXES = 21

IMAGE_OUTPUTS = set([OutputTypes.IMAGE, 
                    OutputTypes.IMAGE_MASKED, 
                    OutputTypes.IMAGE_CROPPED, 
                    OutputTypes.IMAGE_MASKED_CROPPED]
                    )
DEPTH_OUTPUTS = set([OutputTypes.DEPTH_IMAGE,
                    OutputTypes.DEPTH_IMAGE_MASKED, 
                    OutputTypes.DEPTH_IMAGE_MASKED, 
                    OutputTypes.DEPTH_IMAGE_MASKED, 
                    OutputTypes.DEPTH_IMAGE_MASKED, 
                    OutputTypes.DEPTH_IMAGE_MASKED,
                    OutputTypes.DEPTH_POINTS,
                    OutputTypes.DEPTH_POINTS_MASKED,
                    OutputTypes.DEPTH_POINTS_AND_INDEXES,
                    OutputTypes.DEPTH_POINTS_MASKED_AND_INDEXES]
                    )
TRANSFORM_OUTPUTS = set([OutputTypes.TRANSFORM_MATRIX,
                        OutputTypes.ROTATION_MATRIX,
                        OutputTypes.QUATERNION,
                        OutputTypes.TRANSLATION]
                        )
DEPTH_POINT_OUTPUTS = set([OutputTypes.DEPTH_POINTS,
                          OutputTypes.DEPTH_POINTS_MASKED,
                          OutputTypes.DEPTH_POINTS_AND_INDEXES,
                          OutputTypes.DEPTH_POINTS_MASKED_AND_INDEXES]
                          )
POINT_INDEX_OUTPUTS = set([OutputTypes.DEPTH_POINTS_AND_INDEXES,
                          OutputTypes.DEPTH_POINTS_MASKED_AND_INDEXES]
                          )
MODEL_POINT_OUTPUTS = set([OutputTypes.MODEL_POINTS,
                          OutputTypes.MODEL_POINTS_TRANSFORMED]
                          )

MASK_OUTPUTS = set([OutputTypes.MASK, 
                   OutputTypes.IMAGE_MASKED, 
                   OutputTypes.IMAGE_MASKED_CROPPED, 
                   OutputTypes.DEPTH_IMAGE_MASKED, 
                   OutputTypes.DEPTH_IMAGE_MASKED_CROPPED, 
                   OutputTypes.DEPTH_POINTS_MASKED,
                   OutputTypes.DEPTH_POINTS_MASKED_AND_INDEXES]
                   )

BBOX_OUTPUTS = set([OutputTypes.BBOX, 
                   OutputTypes.IMAGE_CROPPED, 
                   OutputTypes.IMAGE_MASKED_CROPPED, 
                   OutputTypes.DEPTH_IMAGE_CROPPED, 
                   OutputTypes.DEPTH_IMAGE_MASKED_CROPPED,
                   OutputTypes.DEPTH_POINTS_MASKED,
                   OutputTypes.DEPTH_POINTS_MASKED_AND_INDEXES]
                   )

CAMERA_MATRIX_OUTPUTS = set([OutputTypes.DEPTH_POINTS,
                            OutputTypes.DEPTH_POINTS_MASKED,
                            OutputTypes.DEPTH_POINTS_AND_INDEXES,
                            OutputTypes.DEPTH_POINTS_MASKED_AND_INDEXES]
                            )


def processImage(img, meta_data, output_type, 
                 remove_mask = True,
                 background_fill = 255., 
                 boarder_width = 0.0,
                 ):
    if(output_type in MASK_OUTPUTS):
        img = np.concatenate([img, meta_data['mask']], axis=2)
        if(self.background_fill is not None):
            image = transparentOverlay(img, background_fill, remove_mask=remove_mask)
    if(output_type in BBOX_OUTPUTS):
        img, _ = cropBBox(img, meta_data['bbox'], boarder_width)
    return torch.from_numpy(np.transpose(img, (2,0,1)).astype(np.float32))

def processDepthImage(depth, meta_data, output_type, 
                      num_points = 1000,
                      boarder_width = 0.0,
                      ):
    if(output_type in MASK_OUTPUTS):
        depth = ma.masked_array(depth, ~np.bitwise_and(meta_data['mask'], depth != 0))

    if(output_type in BBOX_OUTPUTS):
        depth, corner = cropBBox(depth, meta_data['bbox'], boarder_width)
    else:
        corner = (0,0)

    if(output_type in DEPTH_POINT_OUTPUTS):
        y_size, x_size = depth.shape[:2]
        ymap, xmap = np.meshgrid(np.arange(x_size), np.arange(y_size))
        ymap += corner[0]
        xmap += corner[1]
        choose = depth.flatten().nonzero()[0]
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        elif len(choose) > 0:
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
        else:
            raise PoseDataError('No points in mask')
        
        depth_choose = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_choose = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_choose = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        z = depth_choose / meta_data['camera_scale']
        x = (ymap_choose - meta_data['camera_cx']) * z / meta_data['camera_fx']
        y = (xmap_choose - meta_data['camera_cy']) * z / meta_data['camera_fy']
        cloud = np.concatenate((x, y, z), axis=1)
        cloud = torch.from_numpy(cloud.astype(np.float32))
        if(output_type in POINT_INDEX_OUTPUTS):
            return [cloud, torch.LongTensor(choose.astype(np.int32))]
        else:
            return cloud
    
    return torch.from_numpy(depth.astype(np.float32))

def processModelPoints(points, meta_data, output_type):
    if(output_type is OutputTypes.MODEL_POINTS_TRANSFORMED):
        points = points
        R = meta_data['transform_mat'][:3,:3]
        t = meta_data['transform_mat'][:3,3]
        points = np.add(np.dot(points, R.T), t)
    return torch.from_numpy(points.astype(np.float32))

def processTransform(meta_data, output_type): 
    if(output_type is OutputTypes.ROTATION_MATRIX):
        return torch.from_numpy(meta_data['transform_mat'][:3,:3].astype(np.float32))
    if(output_type is OutputTypes.QUATERNION):
        R = np.eye(4)
        R[:3,:3] = meta_data['transform_mat'][:3,:3]
        return torch.from_numpy(quaternion_from_matrix(R).astype(np.float32))
    if(output_type is OutputTypes.TRANSLATION):
        return torch.from_numpy(meta_data['transform_mat'][:3,3].astype(np.float32))
    return torch.from_numpy(meta_data['transform_mat'].astype(np.float32))


class PoseDataset(Dataset):
    def __init__(self, 
                 output_data = [
                                OutputTypes.IMAGE_CROPPED,
                                #OutputTypes.QUATERNION,
                               ],
                 preprocessor = None,
                 image_size = [640, 480],
                 num_points = -1,
                 resample_on_error = False,
                 *args, **kwargs):
        super(PoseDataset, self).__init__()
        self.image_size = image_size
        self.output_data = output_data
        self.output_types = set(output_data)
        
        self.output_data_buffered = []
        for ot in self.output_data:
            self.output_data_buffered.append(ot)
            if(ot in POINT_INDEX_OUTPUTS):
                self.output_data_buffered.append(None)
        
        self.preprocessor = preprocessor
        self.REMOVE_MASK = True
        self.backgroud_fill = 255.0
        self.boarder_width = 0.0
        self.IMAGE_CONTAINS_MASK = False
        self.BBOX_FROM_MASK = False
        self.num_points = num_points
        self.resample_on_error = resample_on_error

    def __getitem__(self, index):
        outputs = [] 
        need_mask = not self.IMAGE_CONTAINS_MASK and len(self.output_types & MASK_OUTPUTS) > 0
        need_bbox = not self.BBOX_FROM_MASK and len(self.output_types & BBOX_OUTPUTS) > 0
        need_camera_matrix = len(self.output_types & CAMERA_MATRIX_OUTPUTS) >  0
        try:
            meta_data = self.getMetaData(index, mask = need_mask, bbox = need_bbox, camera_matrix = need_camera_matrix)
            if(need_mask and self.IMAGE_CONTAINS_MASK):
                img = self.getImage(index)
                meta_data['mask'] = img[:,:,3]
            # Could build a output function at init to speed this up if its slow.
            for output_type in self.output_data:
                if(output_type in IMAGE_OUTPUTS):
                    if('img' not in locals()):
                        img = self.getImage(index)
                    outputs.append(processImage(img.copy(), meta_data, output_type,
                                                remove_mask = self.REMOVE_MASK,
                                                background_fill = self.background_fill,
                                                boarder_width = self.boarder_width))
                elif(output_type in DEPTH_OUTPUTS):
                    if('depth' not in locals()):
                        depth = self.getDepthImage(index)
                    outputs.extend(processDepthImage(depth.copy(), meta_data, output_type,
                                                     num_points = self.num_points,
                                                     boarder_width = self.boarder_width))
                elif(output_type in MODEL_POINT_OUTPUTS):
                    if('points' not in locals()):
                        points = self.getModelPoints(meta_data['object_label'])
                    outputs.append(processModelPoints(points.copy(), meta_data, output_type))
                elif(output_type in TRANSFORM_OUTPUTS):
                    outputs.append(processTransform(meta_data, output_type))
                elif(output_type is OutputTypes.OBJECT_LABEL):
                    outputs.append(torch.LongTensor([meta_data['object_label']]))
                elif(output_type is OutputTypes.MASK):
                    outputs.append(torch.LongTensor(meta_data['mask']))
                elif(output_type is OutputTypes.BBOX):
                    outputs.append(torch.LongTensor(meta_data['bbox']))
                else:
                    raise ValueError('Invalid Output Type {}'.format(output_type))
        except PoseDataError as e:
            print('Exception on index {}: {}'.format(index, e))
            if(self.resample_on_error):
                return self.__getitem__(np.random.randint(0, len(self)))
            else:
                return [[] for _ in self.output_data_buffered]

        if(self.preprocessor is not None):
            outputs = self.preprocessor(outputs, meta_data, self.output_data_buffered)
            
        return tuple(outputs)
    def getDepthImage(self, index):
        raise NotImplementedError('getDepthImage must be implemented by child classes')
    def getImage(self, index):
        raise NotImplementedError('getImage must be implemented by child classes')

    ### Should return dictionary containing {transform_mat, object_label}
    # Optionally containing {mask, bbox, camera_scale, camera_cx, camera_cy, camera_fx, camera_fy}
    def getMetaData(self, index, mask=False, bbox=False, camera_matrix=False):
        raise NotImplementedError('getModelPoints must be implemented by child classes')
    
    def getModelPoints(self, object_label):
        raise NotImplementedError('getModelPoints must be implemented by child classes')
    
    def __len__(self):
        raise NotImplementedError('__len__ must be implemented by child classes')

