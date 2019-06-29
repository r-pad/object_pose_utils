from object_pose_utils.utils.temporal_filtering_framework import TemporalFilteringFramework
from object_pose_utils.utils.multi_view_utils import update_function_gaussian, update_function_bingham
import torch
import numpy as np
from dense_fusion.network import PoseNet
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset as YCBVideoDataset
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.image_processing import ImageNormalizer
from generic_pose.utils import to_np, to_var
from object_pose_utils.utils.pose_processing import tensorAngularDiff

dataset_root = '/home/mengyunx/DenseFusion/datasets/ycb/YCB_Video_Dataset/'
mode = 'train'
object_id = 1
video_id = '0001'
model_checkpoint = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'
output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
                 otypes.IMAGE_CROPPED,
                 otypes.MODEL_POINTS_TRANSFORMED,
                 otypes.MODEL_POINTS,
                 otypes.OBJECT_LABEL,
                 otypes.QUATERNION]

num_objects = 21 #number of object classes in the dataset
num_points = 1000 #number of points on the input pointcloud

ycb_dataset = YCBDataset(dataset_root, mode=mode,
                     object_list = [object_id],
                     output_data = output_format,
                     resample_on_error = True,
                     #preprocessors = [YCBOcclusionAugmentor(dataset_root), ColorJitter()],
                     #postprocessors = [ImageNormalizer(), PointShifter()],
                     postprocessors = [ImageNormalizer()],
                     image_size = [640, 480], num_points=1000)

print("YCB dataset constructed")

ycb_video_dataset = YCBVideoDataset(ycb_dataset, 0, 100)

print("YCB video dataset constructed")

dataloader = torch.utils.data.DataLoader(ycb_video_dataset, batch_size=1, shuffle=False, num_workers=0)
dataloader.dataset.setObjectId(object_id)
dataloader.dataset.setVideoId(video_id)

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.load_state_dict(torch.load(model_checkpoint))
estimator.cuda()

print("Estimator constructed")

update_function = update_function_gaussian

framework = TemporalFilteringFramework(dataloader, estimator, update_function)

print("Intial state: {0}".format(framework.current_rotation))
gt = framework.frames_gt[framework.get_frame_number()]

distance = to_np(tensorAngularDiff(torch.tensor(framework.current_rotation).cuda(), gt.unsqueeze(0).cuda()))*180/np.pi
print("Initial distance (degrees): {0}".format(distance))
count_framework_better = 0
count_densefusion_better = 0 
for i in range(1, dataloader.dataset.__len__()):
    result = framework.propogate()
    if result == 1:
        print("The update has stopped.")
        break
    print("Current frame: {0}\n, Updated state: {1}".format(i,framework.current_rotation))
    
    gt = framework.frames_gt[framework.get_frame_number()]

    distance = to_np(tensorAngularDiff(torch.tensor(framework.current_rotation).float().cuda(), gt.unsqueeze(0).cuda()))*180/np.pi

    print("Distance via framework(degrees): {0}".format(distance))
    distance_densefusion = to_np(tensorAngularDiff(torch.tensor(framework.measure(framework.get_frame_number())).float().cuda(), gt.unsqueeze(0).cuda()))*180/np.pi
    print("Distance via densefusion(degrees): {0}".format(distance_densefusion))
    if distance_densefusion > distance:
        count_framework_better += 1
    elif distance_densefusion < distance:
        count_densefusion_better += 1
    if i == 100:
        break

print("Framework performs better {0} times, and densefusion performs better {1} times".format(count_framework_better, count_densefusion_better))
