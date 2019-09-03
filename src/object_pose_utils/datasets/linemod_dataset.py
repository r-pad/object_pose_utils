from .pose_dataset import PoseDataset
import numpy as np
import random
import numpy.ma as ma
from PIL import Image
import yaml

class LinemodDataset(PoseDataset):
    def __init__(self, dataset_root, mode, object_list, segnet_mask = False,
                 *args, **kwargs):
        super(LinemodDataset, self).__init__(*args, **kwargs)
        self.mode = mode
        self.dataset_root = dataset_root
        self.object_list = object_list
        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.pt = {}
        self.num_pt_mesh_small = 500
        self.list_rank = []
        self.meta = {}
        self.object_labels = [1,2,4,5,6,8,9,10,11,12,13,14,15]
        txt_list = []
        if self.mode == "train":
            for item in object_list: # txt list will be (obj1, obj1, obj1,...,obj2, obj2, obj2,...)
                txt_list.append(('{0}/data/{1}/train.txt'.format(self.dataset_root, '%02d' % item), item))
                self.pt[item] = self.ply_vtx('{0}/models/obj_{1}.ply'.format(self.dataset_root, '%02d' % item))
        else:
            for item in object_list:  # txt list will be (obj1, obj1, obj1,...,obj2, obj2, obj2,...)
                txt_list.append(('{0}/data/{1}/test.txt'.format(self.dataset_root, '%02d' % item), item))
                self.pt[item] = self.ply_vtx('{0}/models/obj_{1}.ply'.format(self.dataset_root, '%02d' % item))
        #print("linemod txt list: {0}".format(len(txt_list)))
        item_count = 0
        recorded_count = 0
        for txt, item in txt_list:
            input_file = open(txt)
            while 1:
                input_line = input_file.readline()
                item_count += 1
                
                if not input_line:
                    break
                if self.mode == 'test' and item_count % 10 != 0:
                    continue

                recorded_count += 1
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.dataset_root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.dataset_root, '%02d' % item, input_line))
                if self.mode == 'eval' or segnet_mask:
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.dataset_root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.dataset_root, '%02d' % item, input_line))
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))
            #print("linemod object {0} has {1}".format(txt, recorded_count))
            input_file.close()
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.dataset_root, '%02d' % item), 'r')
            self.meta[item] = yaml.load(meta_file)
        #print("linemod: {0}".format(item_count))
        image_size = np.array(Image.open(self.list_rgb[0])).shape

    def getDepthImage(self, index):
        path = self.list_depth[index]
        image = np.array(Image.open(path))/1000
        return image

    def getImage(self, index):
        path = self.list_rgb[index]
        image = np.array(Image.open(path))
        return image

    def getModelPoints(self, object_label):
        model_points = self.pt[object_label] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)

        return model_points

    ### Should return dictionary containing {transform_mat, object_label}
    # Optionally containing {mask, bbox, camera_scale, camera_cx, camera_cy, camera_fx, camera_fy}
    def getMetaData(self, index, mask=False, bbox=False, camera_matrix=False):
        returned_dict = {}
        object_label = self.list_obj[index]
        returned_dict['object_label'] = object_label
        # Transform matrix should be in 4x4 format
        rank = self.list_rank[index]

        if object_label == 2:
            for i in range(0, len(self.meta[object_label][rank])):
                if self.meta[object_label][rank][i]['obj_id'] == 2:
                    meta = self.meta[object_label][rank][i]
                    break
        else:
            meta = self.meta[object_label][rank][0]


        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c'])

        transform_mat = np.identity(4)
        transform_mat[:3, :3] = target_r
        transform_mat[:3, 3] = target_t/1000.0

        returned_dict['transform_mat'] = transform_mat

        if mask:
            path = self.list_depth[index]
            depth = self.getDepthImage(index)
            label = np.array(Image.open(self.list_label[index]))
            if(len(label.shape) > 2):
                label = label[:,:,0]

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))

            mask = mask_label * mask_depth
            returned_dict['mask'] = mask

        if bbox: # needs to return x,y,w,h
            bbox = get_bbox(meta['obj_bb'])
            returned_dict['bbox'] = bbox

        if camera_matrix:
            returned_dict['camera_scale'] = 1.0
            returned_dict['camera_cx'] = 325.26110
            returned_dict['camera_cy'] = 242.04899
            returned_dict['camera_fx'] = 572.41140
            returned_dict['camera_fy'] = 573.57043

        return returned_dict


    def ply_vtx(self,path):
        f = open(path)
        assert f.readline().strip() == "ply"
        f.readline()
        f.readline()
        N = int(f.readline().split()[-1])
        while f.readline().strip() != "end_header":
            continue
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
        return np.array(pts)

    def __len__(self):
        return len(self.list_rgb)

    
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
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
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return cmin, rmin, cmax-cmin, rmax-rmin