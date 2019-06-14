from .pose_dataset import PoseDataset
from object_pose_utils.datasets.ycb_dataset import YcbDataset
import os

class YcbVideoDataset(PoseDataset):
    def __init__(self, dataset_root, mode, object_list, 
                 interval, video_num,
                 use_label_bbox = True, grid_size = 3885, 
                 add_syn_background = True,
                 background_files = None,
                 add_syn_noise = True,
                 refine = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.dataset_root = dataset_root
        self.object_list = object_list
        self.index_list = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.pt = {}
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.minimum_num_pts = 50
        self.list_rank = []
        self.index_to_object_name = {} #{2:002_master_chef_can, 3:...}
        self.interval = interval
        self.use_label_bbox = use_label_bbox
        self.add_syn_background = add_syn_background
        self.add_syn_noise = add_syn_noise
        self.refine = refine
        self.classes = ['__background__']
        self.video_num = video_num

        self.ycb_dataset = YcbDataset(dataset_root, mode, object_list, use_label_bbox, grid_size,add_syn_background, background_files, add_syn_noise, refine, args, kwargs)

        # Fill in the index to object name array
        with open(os.path.join(self.dataset_root, 'YCB_Video_Dataset/image_sets', 'classes.txt')) as f:
            self.classes.extend([x.rstrip('\n') for x in f.readlines()])
        
        # Collect corresponding images
        # image_list will contain (0086/000867, object num), (0086/000868, object num) ...
        local_index = 0
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
                    # If the frame is within the correct video sequence
                    if image_line[:4] == format(self.video_num, "04"):
                        self.index_list.append((local_index, item))
                        self.list_obj.append(item)
                    local_index += 1
                image_file.close()
        self.len_real = len(self.index_list)

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
                    if image_line[:4] == format(self.video_num, "04"):
                        self.index_list.append((local_index, item))
                        self.list_obj.append(item)
                    local_index += 1
                image_file.close()
        self.len_syn = len(self.index_list) - self.len_real
        
        if 'grid' in mode:
            num_digits = len(str(grid_size))
            for item in object_list:
                for j in range(grid_size):
                    image_line = '../depth_renders/{1}/{2:0{0}d}'.format(num_digits, self.classes[item], j)
                    if image_line[:4] == format(self.video_num, "04"):
                        self.index_list.append((local_index, item))
                        self.list_obj.append(item)
                    local_index += 1
        self.len_grid = len(self.index_list) - self.len_real - self.len_syn

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
                    if image_line[:4] == format(self.video_num, "04"):
                        self.index_list.append((local_index, item))
                        self.list_obj.append(item)
                    local_index += 1
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
                    if image_line[:4] == format(self.video_num, "04"):
                        self.index_list.append((local_index, item))
                        self.list_obj.append(item)
                    local_index += 1
                image_file.close()


    def getDepthImage(self, index):
        image_list = []
        current_index = index
        while current_index < len(self.index_list):
            depth_image = self.ycb_dataset.getDepthImage(current_index)
            image_list.extend(depth_image)
            current_index += self.interval
        return image_list

    def getImage(self, index):
        image_list = []
        current_index = index
        while current_index < len(self.index_list):
            color_image = self.ycb_dataset.getImage(current_index)
            image_list.extend(color_image)
            current_index += self.interval
        print("image_list size {0}".format(len(image_list)))
        return image_list

    ### Should return dictionary containing {transform_mat, object_label}                                                                
    # Optionally containing {mask, bbox, camera_scale, camera_cx, cam\era_cy, camera_fx, camera_fy}
    def getMetaData(self, index, mask=False, bbox=False, camera_matrix=False):
        metadata_list = []
        current_index = index
        while current_index < len(self.index_list):
            metadata = self.ycb_dataset.getMetaData(index, mask, bbox, camera_matrix)
            metadata_list.extend(metadata)
            current_index += self.interval
        return metadata_list

    # Note: object_label here needs to be consistent to the object label returned in getMetaData
    def getModelPoints(self, object_label):
        model_points = self.ycb_dataset.getModelPoints(object_label)
        return model_points

    def __len__(self):
        print("index_list has {0} entries".format(len(self.index_list)))
        print("Interval is {0}".format(self.interval))
        return len(self.index_list) // self.interval
