# A wrapper class for ycb_dataset.py that supports:
# Given a video dataset, n = number of frames to sample, k = interval between sampled frames
# Output information about the sampled frames (color images, depth images, metadata, etc) with a specified start frame d.  

from .pose_dataset import PoseDataset
from object_pose_utils.datasets.ycb_dataset import YcbDataset
import os
import scipy.io as scio
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.image_processing import ImageNormalizer

class YcbVideoDataset:
    def __init__(self, dataset_root, mode, object_id, 
                 interval, video_len, video_num,
                 use_label_bbox = True, grid_size = 3885, 
                 add_syn_background = True,
                 background_files = None,
                 add_syn_noise = True,
                 refine = False, *args, **kwargs):
        self.mode = mode
        self.dataset_root = dataset_root
        self.object_id = object_id
        self.index_list = []
        self.index_to_object_name = {} #{2:002_master_chef_can, 3:...}
        self.interval = interval
        self.use_label_bbox = use_label_bbox
        self.add_syn_background = add_syn_background
        self.add_syn_noise = add_syn_noise
        self.refine = refine
        self.class_name = ''
        self.video_num = video_num
        self.video_len = video_len
        
        # Initialize the wrapped ycb_dataset
        output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
                                          otypes.IMAGE_CROPPED,
                                          otypes.MODEL_POINTS_TRANSFORMED,
                                          otypes.MODEL_POINTS,
                                          otypes.OBJECT_LABEL,
                                          otypes.QUATERNION,
                         ]
        self.ycb_dataset = YcbDataset(dataset_root, mode=mode,
                                                  object_list = [object_id],
                                                  output_data = output_format,
                                                  resample_on_error = True,
                                                  #preprocessors = [YCBOcclusionAugmentor(dataset_root), ColorJitter()],
                                                  #postprocessors = [ImageNormalizer(), PointShifter()],
                                                  postprocessors = [ImageNormalizer()],
                                                  image_size = [640, 480], num_points=1000)
        #self.ycb_dataset = YcbDataset(dataset_root, mode, [object_id], use_label_bbox, grid_size,add_syn_background, background_files, add_syn_noise, refine, args, kwargs)

        # Fill in the object id to class name list
        classes = ['__background__']
        with open(os.path.join(self.dataset_root, 'YCB_Video_Dataset/image_sets', 'classes.txt')) as f:
            classes.extend([x.rstrip('\n') for x in f.readlines()])
        self.class_name = classes[self.object_id]

        print("class name: {0}".format(self.class_name))
        # Collect corresponding global indexes
        # index_list will contain the indexes to call in ycb_dataset
        local_index = 0
        if 'train' in mode:
            path = '{0}/YCB_Video_Dataset/image_sets/{1}_train_split.txt'.format(self.dataset_root, self.class_name)
            file_id_file = open(path)
            while 1:
                file_id_line = file_id_file.readline()
                if not file_id_line:
                    break
                # If the frame is within the correct video
                if file_id_line[:4] == format(self.video_num, "04"):
                    self.index_list.append(local_index)
                local_index += 1
            file_id_file.close()
        self.len_real = len(self.index_list)

        if 'syn' in mode:
            path = '{0}/YCB_Video_Dataset/image_sets/{1}_syn.txt'.format(self.dataset_root, self.class_name)
            image_file = open(path)
            while 1:
                image_line = image_file.readline()
                if not image_line:
                    break
                if image_line[:4] == format(self.video_num, "04"):
                    self.index_list.append(local_index)
                local_index += 1
            image_file.close()
        self.len_syn = len(self.index_list) - self.len_real
        
        if 'grid' in mode:
            num_digits = len(str(grid_size))
            for j in range(grid_size):
                image_line = '../depth_renders/{1}/{2:0{0}d}'.format(num_digits, self.class_name, j)
                if image_line[:4] == format(self.video_num, "04"):
                    self.index_list.append(local_index)
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
            path = '{0}/YCB_Video_Dataset/image_sets/{1}_valid_split.txt'.format(self.dataset_root, self.class_name)
            image_file = open(path)
            while 1:
                image_line = image_file.readline()
                if not image_line:
                    break
                if image_line[:4] == format(self.video_num, "04"):
                    self.index_list.append(local_index)
                local_index += 1
            image_file.close()                               

        if mode == 'test':
            path = '{0}/YCB_Video_Dataset/image_sets/{1}_keyframe.txt'.format(self.dataset_root, self.class_name)
            image_file = open(path)
            while 1:
                image_line = image_file.readline()
                if not image_line:
                    break
                if image_line[:4] == format(self.video_num, "04"):
                    self.index_list.append(local_index)
                local_index += 1
            image_file.close()

    # Return a list of depth images corresponding to the sampled subvideo
    def getDepthImage(self, index):
        # Input: index wrt local index in the specifide video
        
        image_list = []
        current_index = index
        count = 0
        while current_index < len(self.index_list) and count < self.video_len:
            global_index = self.index_list[current_index]
            depth_image = self.ycb_dataset.getDepthImage(global_index)
            image_list.append(depth_image)
            count += 1
            current_index += self.interval
        return image_list

    def getImage(self, index):
        # Input: index wrt local index in the specifide video
        image_list = []
        current_index = index
        count = 0 
        while current_index < len(self.index_list) and count < self.video_len:
            global_index = self.index_list[current_index]
            color_image = self.ycb_dataset.getImage(global_index)
            image_list.append(color_image)
            count += 1
            current_index += self.interval
        return image_list

    ### Should return dictionary containing {transform_mat, object_label}                                                                
    # Optionally containing {mask, bbox, camera_scale, camera_cx, cam\era_cy, camera_fx, camera_fy}
    def getMetaData(self, index, mask=False, bbox=False, camera_matrix=False):
        # Input: index wrt local index in the specifide video
        metadata_list = []
        current_index = index
        count = 0
        while current_index < len(self.index_list) and count < self.video_len:
            global_index = self.index_list[current_index]
            metadata = self.ycb_dataset.getMetaData(global_index, mask, bbox, camera_matrix)
            metadata_list.append(metadata)
            count += 1
            current_index += self.interval
        return metadata_list

    # Note: object_label here needs to be consistent to the object label returned in getMetaData
    def getModelPoints(self):
        model_points = self.ycb_dataset.getModelPoints(self.object_id)
        return model_points

    def getItem(self, index):
        # Input: index wrt local index in the specifide video
        outputs_list = []
        current_index = index
        count = 0 
        while current_index < len(self.index_list) and count < self.video_len:
            global_index = self.index_list[current_index]
            outputs = self.ycb_dataset.__getitem__(global_index)
            outputs_list.append(outputs)
            count += 1
            current_index += self.interval
        return outputs_list

    def getCameraTransforms(self, index):
        transform_list = []
        current_index = index
        count = 0
        while current_index < len(self.index_list) and count < self.video_len:
            global_index = self.index_list[current_index]
            sub_path = self.ycb_dataset.getPath(global_index)
            path = '{0}/YCB_Video_Dataset/data/{1}-meta.mat'.format(self.dataset_root, sub_path)
            meta = scio.loadmat(path)
            rotation_translation_matrix = meta["rotation_translation_matrix"]
            transform_list.append(rotation_translation_matrix)
            current_index += self.interval
            count += 1

        return transform_list

    def getLen(self):
        return len(self.index_list)
