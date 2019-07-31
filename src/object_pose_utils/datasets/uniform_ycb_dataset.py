# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:38:19 2018

@author: bokorn
"""
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import os
import pickle

from object_pose_utils.datasets.pose_dataset import PoseDataError

from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.ycb_dataset import get_bbox_label

class UniformYCBDataset(YCBDataset):
    def __init__(self, dataset_root, mode, object_label, use_label_bbox = True, fill_with_exact = True,
            *args, **kwargs):
        super(YCBDataset, self).__init__(*args, **kwargs)

        self.add_val = 'valid' in mode
        self.dataset_root = dataset_root
        self.use_label_bbox = use_label_bbox
        self.minimum_num_pts = 50
        self.fill_with_exact = fill_with_exact
        self.classes = ['__background__']

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189


        # Fill in the index to object name array
        with open(os.path.join(self.dataset_root, 'image_sets', 'classes.txt')) as f:
            self.classes.extend([x.rstrip('\n') for x in f.readlines()])

        # Possibly allow for randomly selecting object. 
        # For now its one object per dataset and well use concat dataset
        self.setObject(object_label)

        with open(os.path.join(self.dataset_root, 'image_sets', 'binned_grid_offset.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.offset_tetra_bins = data['syn_tetra_bins']

        if(fill_with_exact):
            with open(os.path.join(self.dataset_root, 'image_sets', 'binded_grid.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.vert_tetra_bins = data['syn_tetra_bins']

    def setObject(self, object_label):
        self.object_label = object_label
        with open(os.path.join(self.dataset_root, 'image_sets', 'binded_files.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.tetra_bins = data['tetra_bins'][object_label]
        if(self.add_val):
            with open(os.path.join(self.dataset_root, 'image_sets', 'binded_files_valid.pkl'), 'rb') as f:
                val_data = pickle.load(f)
                val_bins = val_data['tetra_bins'][object_label]
                for k in range(len(self.tetra_bins)):
                    self.tetra_bins[k].extend(val_bins[k])

    def getPath(self, index):
        if(len(self.tetra_bins[index])):
            sub_path = np.random.choice(self.tetra_bins[index])
        elif(len(self.offset_tetra_bins[index])):
            render_idx = np.random.choice(self.offset_tetra_bins[index])
            sub_path = 'depth_renders_offset/{0}/{1}'.format(self.classes[self.object_label], render_idx)
        elif(self.fill_with_exact):
            render_idx = np.random.choice(self.vert_tetra_bins[index])
            sub_path = 'depth_renders/{0}/{1}'.format(self.classes[self.object_label], render_idx)
        else:
            raise PoseDataError('No data for bin {} and fill_with_exact not set'.format(index))

        return sub_path

    ### Should return dictionary containing {transform_mat, object_label}
    # Optionally containing {mask, bbox, camera_scale, camera_cx, camera_cy, camera_fx, camera_fy}
    def getMetaData(self, index, mask=False, bbox=False, camera_matrix=False):
        sub_path = '../' + self.getPath(index)
        syn_data = sub_path[:13] == 'depth_renders'  

        returned_dict = {}
        returned_dict['object_label'] = self.object_label

        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.dataset_root, sub_path))

        pose_idx = np.where(meta['cls_indexes'].flatten()==self.object_label)[0][0]
        target_r = meta['poses'][:, :, pose_idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, pose_idx][:, 3:4].flatten()])

        transform_mat = np.identity(4)
        transform_mat[:3, :3] = target_r
        transform_mat[:3, 3] = target_t

        returned_dict['transform_mat'] = transform_mat
        if mask or (bbox and self.use_label_bbox):
            obj = meta['cls_indexes'].flatten().astype(np.int32)
            depth = self.getDepthImage(index)
            path = '{0}/{1}-label.png'.format(self.dataset_root, sub_path)
            label = np.array(Image.open(path))
            
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, self.object_label))
            mask = mask_label * mask_depth

            #TODO: figure out how to handle when the valid labels are smaller than minimum size required
            if len(mask.nonzero()[0]) <= self.minimum_num_pts:
                
                raise PoseDataError('Mask {} has less than minimum number of pixels ({} < {})'.format(
                    sub_path, len(mask.nonzero()[0]), self.minimum_num_pts))
                #while 1:
                    #pass
            returned_dict['mask'] = mask
        if bbox:  # needs to return x,y,w,h
            if(self.use_label_bbox or syn_data):
                rmin, rmax, cmin, cmax = get_bbox_label(mask_label, image_size = self.image_size)
            else:
                posecnn_meta = scio.loadmat('{0}/{1}-posecnn.mat'.format(self.dataset_root, sub_path))
                obj_idx = np.nonzero(posecnn_meta['rois'][:,1].astype(int) == self.object_label)[0]
                if(len(obj_idx) == 0):
                    raise PoseDataError('Object {} not in PoseCNN ROIs {}'.format(self.object_label, sub_path))
                obj_idx = obj_idx[0]
                rois = np.array(posecnn_meta['rois'])
                rmin, rmax, cmin, cmax = self.get_bbox(rois[obj_idx])
            returned_dict['bbox'] = (cmin, rmin, cmax-cmin, rmax-rmin)

        if camera_matrix:
            if not syn_data and sub_path[:8] != 'data_syn' and int(sub_path[5:9]) >= 60:
                cam_cx = self.cam_cx_2
                cam_cy = self.cam_cy_2
                cam_fx = self.cam_fx_2
                cam_fy = self.cam_fy_2
            else:
                cam_cx = self.cam_cx_1
                cam_cy = self.cam_cy_1
                cam_fx = self.cam_fx_1
                cam_fy = self.cam_fy_1

            cam_scale = meta['factor_depth'][0][0]

            returned_dict['camera_scale'] = cam_scale
            returned_dict['camera_cx'] = cam_cx
            returned_dict['camera_cy'] = cam_cy
            returned_dict['camera_fx'] = cam_fx
            returned_dict['camera_fy'] = cam_fy

        return returned_dict

    def __len__(self):
        return len(self.tetra_bins)


