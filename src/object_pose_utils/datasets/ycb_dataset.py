# -*- coding: utf-8 -*-
"""
Created on a Thursday long long ago

@author: bokorn
"""

from .pose_dataset import PoseDataset
import numpy as np
import random
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import os

from object_pose_utils.datasets.pose_dataset import PoseDataError
from object_pose_utils.datasets.image_processing import get_bbox_label

class YcbDataset(PoseDataset):
    def __init__(self, dataset_root, mode, object_list, 
                 use_label_bbox = True, grid_size = 3885, 
                 add_syn_background = True,
                 background_files = None,
                 add_syn_noise = True,
                 refine = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.dataset_root = dataset_root
        self.object_list = object_list
        self.image_list = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.pt = {}
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.minimum_num_pts = 50
        self.list_rank = []
        self.index_to_object_name = {} #{2:002_master_chef_can, 3:...}
        self.use_label_bbox = use_label_bbox
        self.add_syn_background = add_syn_background
        self.add_syn_noise = add_syn_noise
        self.refine = refine
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
        with open(os.path.join(self.dataset_root, 'YCB_Video_Dataset/image_sets', 'classes.txt')) as f:
            self.classes.extend([x.rstrip('\n') for x in f.readlines()])

        #now self.classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
        #                '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_cban', '011_banana', '019_pitcher_base', \
        #                '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
        #                '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')

        # Collect corresponding images
        # image_list will contain (0086/000867, object num), (0086/000868, object num) ...
        if 'train' in mode:
            for item in object_list:
                path = '{0}/YCB_Video_Dataset/image_sets/{1}_train_split.txt'.format(self.dataset_root, self.classes[item])
                image_file = open(path)
                while 1:
                    image_line = image_file.readline()
                    if not image_line:
                        break
                    if image_line[-1] == '\n':
                        image_line = image_line[:-1]
                    self.image_list.append((image_line, item))
                    self.list_obj.append(item)

                image_file.close()
        self.len_real = len(self.image_list)

        if 'syn' in mode:
            for item in object_list:
                path = '{0}/YCB_Video_Dataset/image_sets/{1}_syn.txt'.format(self.dataset_root, self.classes[item])
                image_file = open(path)
                while 1:
                    image_line = image_file.readline()
                    if not image_line:
                        break
                    if image_line[-1] == '\n':
                        image_line = image_line[:-1]
                    self.image_list.append((image_line, item))
                    self.list_obj.append(item)
                image_file.close()
        self.len_syn = len(self.image_list)
        
        if 'grid' in mode:
            num_digits = len(str(grid_size))
            for item in object_list:
                for j in range(grid_size):
                    image_line = '../depth_renders/{1}/{2:0{0}d}'.format(num_digits, self.classes[item], j)
                    self.image_list.append((image_line, item))
                    self.list_obj.append(item)

        self.len_grid = len(self.image_list)

        if self.add_syn_background and ('syn' in mode or 'grid' in mode):
            if(background_files is None):
                background_files = '{0}/YCB_Video_Dataset/image_sets/train_split.txt'.format(self.dataset_root)
            with open(background_files) as f:
                self.background = ['{0}/YCB_Video_Dataset/data/{1}-color.png'.format(self.dataset_root, x.rstrip('\n')) for x in f.readlines()]
        else:
            self.background = None

        if 'valid' in mode:
            for item in object_list:
                path = '{0}/YCB_Video_Dataset/image_sets/{1}_valid_split.txt'.format(self.dataset_root, self.classes[item])
                image_file = open(path)
                while 1:
                    image_line = image_file.readline()
                    if not image_line:
                        break
                    if image_line[-1] == '\n':
                        image_line = image_line[:-1]
                    self.image_list.append((image_line, item))
                    self.list_obj.append(item)

                image_file.close()                               

        if mode == 'test':
            for item in object_list:
                path = '{0}/YCB_Video_Dataset/image_sets/{1}_keyframe.txt'.format(self.dataset_root,
                                                                                     self.classes[item])
                image_file = open(path)
                while 1:
                    image_line = image_file.readline()
                    if not image_line:
                        break
                    if image_line[-1] == '\n':
                        image_line = image_line[:-1]
                    self.image_list.append((image_line, item))
                    self.list_obj.append(item)

                image_file.close()

    def getPath(self, index):
        #print("index is {0}".format(index))
        return self.image_list[index][0]

    def getDepthImage(self, index):
        sub_path = self.getPath(index)
        path = '{0}/YCB_Video_Dataset/data/{1}-depth.png'.format(self.dataset_root, sub_path)
        image = np.array(Image.open(path))
        return image

    def getImage(self, index):
        sub_path = self.getPath(index)
        path = '{0}/YCB_Video_Dataset/data/{1}-color.png'.format(self.dataset_root, sub_path)
        image = np.array(Image.open(path))
        if(self.IMAGE_CONTAINS_MASK):
            mask = image[:,:,3:]
        image = image[:,:,:3]
        if(index >= self.len_real and index < self.len_grid):
            if(self.add_syn_background): 
                label = np.expand_dims(np.array(Image.open('{0}/YCB_Video_Dataset/data/{1}-label.png'.format(self.dataset_root, sub_path))), 2)
                mask_back = ma.getmaskarray(ma.masked_equal(label, 0))
                back_filename = random.choice(self.background)
                back = np.array(Image.open(back_filename).convert("RGB"))
                image = back * mask_back + image
            if(self.add_syn_noise):
                image = image + np.random.normal(loc=0.0, scale=7.0, size=image.shape)
                image = image.astype(np.uint8) 
        if(self.IMAGE_CONTAINS_MASK):
            image = np.concatenate([image, mask], 2)
        return image

    ### Should return dictionary containing {transform_mat, object_label}
    # Optionally containing {mask, bbox, camera_scale, camera_cx, camera_cy, camera_fx, camera_fy}
    def getMetaData(self, index, mask=False, bbox=False, camera_matrix=False):
        #print(self.image_list[index])
        returned_dict = {}
        object_label = self.list_obj[index]
        returned_dict['object_label'] = object_label

        sub_path = self.image_list[index][0]
        path = '{0}/YCB_Video_Dataset/data/{1}-meta.mat'.format(self.dataset_root, sub_path)
        meta = scio.loadmat(path)

        pose_idx = np.where(meta['cls_indexes'].flatten()==object_label)[0][0]
        target_r = meta['poses'][:, :, pose_idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, pose_idx][:, 3:4].flatten()])

        transform_mat = np.identity(4)
        transform_mat[:3, :3] = target_r
        transform_mat[:3, 3] = target_t

        returned_dict['transform_mat'] = transform_mat
        if mask or (bbox and self.use_label_bbox):
            obj = meta['cls_indexes'].flatten().astype(np.int32)
            depth = self.getDepthImage(index)
            sub_path = self.image_list[index][0]
            path = '{0}/YCB_Video_Dataset/data/{1}-label.png'.format(self.dataset_root, sub_path)
            label = np.array(Image.open(path))
            
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, self.list_obj[index]))
            mask = mask_label * mask_depth

            #TODO: figure out how to handle when the valid labels are smaller than minimum size required
            if len(mask.nonzero()[0]) <= self.minimum_num_pts:
                
                raise PoseDataError('Mask {} has less than minimum number of pixels ({} < {})'.format(
                    sub_path, len(mask.nonzero()[0]), self.minimum_num_pts))
                #while 1:
                    #pass
            
            returned_dict['mask'] = mask
        if bbox:  # needs to return x,y,w,h
            if(self.use_label_bbox or (index >= self.len_real and index < self.len_grid)):
                bbox = get_bbox_label(mask_label, image_size = self.image_size)
            else:
                posecnn_meta = scio.loadmat('{0}/YCB_Video_Dataset/data/{1}-posecnn.mat'.format(self.dataset_root, sub_path))
                obj_idx = np.nonzero(posecnn_meta['rois'][:,1].astype(int) == object_label)[0]
                if(len(obj_idx) == 0):
                    raise PoseDataError('Object {} not in PoseCNN ROIs {}'.format(object_label, sub_path))
                obj_idx = obj_idx[0]
                rois = np.array(posecnn_meta['rois'])
                bbox = self.get_bbox(rois[obj_idx])
            returned_dict['bbox'] = bbox 

        if camera_matrix:
            if self.image_list[index][0][:3] != '../' and int(self.image_list[index][0][5:9]) >= 60:
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


    # Note: object_label here needs to be consistent to the object label returned in getMetaData
    def getModelPoints(self, object_label):
        object_name = self.classes[object_label]

        input_file = open('{0}/YCB_Video_Dataset/models/{1}/points.xyz'.format(self.dataset_root, object_name))
        cld = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            cld.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        cld = np.array(cld)
        input_file.close()

        dellist = [j for j in range(0, len(cld))]
        if self.refine:
            dellist = random.sample(dellist, len(cld) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(cld) - self.num_pt_mesh_small)
        model_points = np.delete(cld, dellist, axis=0)

        return model_points

    def __len__(self):
        return len(self.image_list)

    def get_bbox(self, rois):
        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        img_width = self.image_size[1]
        img_length = self.image_size[0]
        rmin = int(rois[3]) + 1
        rmax = int(rois[5]) - 1
        cmin = int(rois[2]) + 1
        cmax = int(rois[4]) - 1
        r_b = rmax - rmin
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_width:
            delt = rmax - img_width
            rmax = img_width
            rmin -= delt
        if cmax > img_length:
            delt = cmax - img_length
            cmax = img_length
            cmin -= delt
        return cmin, rmin, cmax-cmin, rmax-rmin

  
                 
"""
Created on Tue April 29 2019

@author: bokorn
"""
def ycbRenderTransform(q):
    trans_quat = q.copy()
    trans_quat = tf_trans.quaternion_multiply(trans_quat, tf_trans.quaternion_about_axis(-np.pi/2, [1,0,0]))
    return viewpoint2Pose(trans_quat)

def setYCBCamera(renderer, width=640, height=480):
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    renderer.setCameraMatrix(fx, fy, px, py, width, height)

def getYCBSymmeties(obj):
    if(obj == 13):
        return [[0,0,1]], [[np.inf]]
    elif(obj == 16):
        return [[0.9789,-0.2045,0.], [0.,0.,1.]], [[0.,np.pi], [0.,np.pi/2,np.pi,3*np.pi/2]]
    elif(obj == 19):
        return [[-0.14142136,  0.98994949,0]], [[0.,np.pi]]
    elif(obj == 20):
        return [[0.9931506 , 0.11684125,0]], [[0.,np.pi]]
    elif(obj == 21):
        return [[0.,0.,1.]], [[0.,np.pi]]
    else:
        return [],[]
