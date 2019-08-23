from .pose_dataset import PoseDataset
import numpy as np
import random
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import os

from object_pose_utils.datasets.pose_dataset import PoseDataError
from object_pose_utils.datasets.image_processing import get_bbox_label

class ShapeNetDataset(PoseDataset):
    def __init__(self, dataset_root, mode, object_list, 
                 use_label_bbox = True, grid_size = 87, 
                 add_syn_background = True,
                 background_files = None,
                 add_syn_noise = True,
                 refine = False,
                 *args, **kwargs):
        super(ShapeNetDataset, self).__init__(*args, **kwargs)
        self.mode = mode
        self.dataset_root = dataset_root
        self.object_list = object_list
        self.image_list = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 1000
        self.minimum_num_pts = 50
        self.use_label_bbox = use_label_bbox
        self.add_syn_background = add_syn_background
        self.add_syn_noise = add_syn_noise
        self.refine = refine
        self.classes = ['__background__']

        self.cam_cx = 312.9869
        self.cam_cy = 241.3109
        self.cam_fx = 1066.778
        self.cam_fy = 1067.487

        num_digits = len(str(grid_size))
        for item in object_list:
            for j in range(grid_size):
                image_line = '{1}/{2}/{3:0{0}d}'.format(num_digits, item[0], item[1], j)
                if not os.path.isfile(dataset_root + '/87grid_renders/' + image_line + '-meta.mat'):
                    continue
                self.image_list.append((image_line, item))
                self.list_obj.append(item)

        # if self.add_syn_background:
        #     if (background_files is None):
        #         background_files = 
        #     with open(background_files)

    def getPath(self, index):
        return self.image_list[index][0]

    def getDepthImage(self, index):
        sub_path = self.getPath(index)
        path = '{0}/87grid_renders/{1}-depth.png'.format(self.dataset_root, sub_path)
        image = np.array(Image.open(path))
        return image

    def getImage(self,index):
        sub_path = self.getPath(index)
        path = '{0}/87grid_renders/{1}-color.png'.format(self.dataset_root, sub_path)
        image = np.array(Image.open(path))
        if(self.IMAGE_CONTAINS_MASK):
            mask = image[:,:,3:]
        image = image[:,:,:3]
        # if(index >= self.len_real and index < self.len_grid):
        if True:
            if(self.add_syn_background): 
                label = np.expand_dims(np.array(Image.open('{0}/87grid_renders/{1}-label.png'.format(self.dataset_root, sub_path))), 2)
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

    def getMetaData(self, index, mask=False, bbox=False, camera_matrix=False):
        returned_dict = {}
        # object_label = self.list_obj[index][0] + '-' + self.list_obj[index][1]
        object_label = '{0}/{1}'.format(self.list_obj[index][0], self.list_obj[index][1])
        returned_dict['object_label'] = object_label
        # returned_dict['object_label'] = 1

        sub_path = self.image_list[index][0]
        path = '{0}/87grid_renders/{1}-meta.mat'.format(self.dataset_root, sub_path)
        meta = scio.loadmat(path)

        pose_idx = meta['cls_indexes'][0][0]
        pose_idx = 0
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
            path = '{0}/87grid_renders/{1}-label.png'.format(self.dataset_root, sub_path)
            label = np.array(Image.open(path))
            
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            # mask_label = ma.getmaskarray(ma.masked_equal(label, self.list_obj[index]))
            mask_label = ma.getmaskarray(ma.masked_equal(label, 1))
            mask = mask_label * mask_depth

            #TODO: figure out how to handle when the valid labels are smaller than minimum size required
            if len(mask.nonzero()[0]) <= self.minimum_num_pts:
                
                raise PoseDataError('Mask {} has less than minimum number of pixels ({} < {})'.format(
                    sub_path, len(mask.nonzero()[0]), self.minimum_num_pts))
                #while 1:
                    #pass
            returned_dict['mask'] = mask

        if bbox:
            bbox = get_bbox_label(mask_label, image_size = self.image_size)
            returned_dict['bbox'] = bbox

        if camera_matrix:
            cam_cx = self.cam_cx
            cam_cy = self.cam_cy
            cam_fx = self.cam_fx
            cam_fy = self.cam_fy

            cam_scale = meta['factor_depth'][0][0]

            returned_dict['camera_scale'] = cam_scale
            returned_dict['camera_cx'] = cam_cx
            returned_dict['camera_cy'] = cam_cy
            returned_dict['camera_fx'] = cam_fx
            returned_dict['camera_fy'] = cam_fy

        return returned_dict

    def getModelPoints(self, object_label):
        #object_name = self.classes[object_label]

        input_file = open('{0}/models/{1}/model.xyz'.format(self.dataset_root, object_label))
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

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    def __len__(self):
        return len(self.image_list)


