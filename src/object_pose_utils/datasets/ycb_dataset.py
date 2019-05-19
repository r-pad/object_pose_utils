from .pose_dataset import PoseDataset
import numpy as np
import random
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import os

class YcbDataset(PoseDataset):
    def __init__(self, dataset_root, mode, object_list):
        super().__init__(0)
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
        self.minimum_num_pt = -1#50
        self.list_rank = []
        self.index_to_object_name = {} #{2:002_master_chef_can, 3:...}
        self.classes = ['__background__']

        # Fill in the index to object name array
        with open(os.path.join(self.dataset_root, 'YCB_Video_Dataset', 'image_sets', 'classes.txt')) as f:
            self.classes.extend([x.rstrip('\n') for x in f.readlines()])

        #now self.classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
        #                '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
        #                '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
        #                '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')

        # Collect corresponding images
        # image_list will contain (0086/000867, object num), (0086/000868, object num) ...
        if mode == 'train':
            for item in object_list:
                path = '{0}/YCB_Video_Dataset/image_sets/{1}_train_split.txt'.format(self.dataset_root,
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

        elif mode == 'test':
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


    def getDepthImage(self, index):
        sub_path = self.image_list[index][0]
        path = '{0}/YCB_Video_Dataset/data/{1}-depth.png'.format(self.dataset_root, sub_path)
        image = np.array(Image.open(path))
        return image

    def getImage(self, index):
        sub_path = self.image_list[index][0]
        path = '{0}/YCB_Video_Dataset/data/{1}-color.png'.format(self.dataset_root, sub_path)
        image = np.array(Image.open(path))
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
        if mask:
            obj = meta['cls_indexes'].flatten().astype(np.int32)
            depth = self.getDepthImage(index)
            sub_path = self.image_list[index][0]
            path = '{0}/YCB_Video_Dataset/data/{1}-label.png'.format(self.dataset_root, sub_path)
            label = np.array(Image.open(path))
            
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, self.list_obj[index]))
            mask = mask_label * mask_depth

            #TODO: figure out how to handle when the valid labels are smaller than minimum size required
            #if len(mask.nonzero()[0]) <= self.minimum_num_pt:
                #print("nonzero in mask is only {0}".format(len(mask.nonzero()[0])))
                #while 1:
                    #pass
            returned_dict['mask'] = mask
        if bbox:  # needs to return x,y,w,h
            obj = meta['cls_indexes'].flatten().astype(np.int32)
            depth = self.getDepthImage(index)
            sub_path = self.image_list[index][0]
            path = '{0}/YCB_Video_Dataset/data/{1}-label.png'.format(self.dataset_root, sub_path)
            label = np.array(Image.open(path))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, self.list_obj[index]))
            mask = mask_label * mask_depth
            #TODO: figure out how to handle when the valid labels are smaller than minimum size required
            #if len(mask.nonzero()[0]) <= self.minimum_num_pt:
                #print("nonzero in mask is only {0}".format(len(mask.nonzero()[0])))
                #while 1:
                    #pass

            rmin, rmax, cmin, cmax = self.get_bbox(mask_label)
            returned_dict['bbox'] = (cmin, rmin, cmax-cmin, rmax-rmin)

        if camera_matrix:
            if self.image_list[index][0][:8] != 'data_syn' and int(self.image_list[index][0][5:9]) >= 60:
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

    def get_bbox(self,label):
        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        img_width = 480
        img_length = 640
        rows = np.any(label, axis=1)
        cols = np.any(label, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
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
            
        return rmin, rmax, cmin, cmax
                 