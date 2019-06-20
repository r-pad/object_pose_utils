import torch
import os
import numpy as np
from object_pose_utils.utils.pose_processing import tensorAngularDiff
from generic_pose.utils import to_np, to_var
from torch.autograd import Variable
# Import video dataloader
from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset
from object_pose_utils.utils.multi_view_utils import applyTransform, computeCameraTransform
from quat_math.rotation_averaging import projectedAverageQuaternion
from quat_math.quat_math import quatAngularDiff
from quat_math import quaternion_from_matrix

# Import estimator
from dense_fusion.evaluate import DenseFusionEstimator
from dense_fusion.data_processing import getYCBGroundtruth
from dense_fusion.network import PoseNet

# Choose the video
video_num = 1
video_path = '/home/mengyunx/DenseFusion/datasets/ycb'
mode = 'train'
object_label = 1
model_checkpoint = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'
num_points = 1000
num_obj = 21


estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.load_state_dict(torch.load(model_checkpoint))
estimator.cuda()

# Choose a sample interval (sample every n frames. For n = 3, samples are [1] 2 3 [4] 5 6 for start frame = 0
sample_interval = 50
sample_num = 10

# For each sequence of samples, apply camera poses to get the final frame's predicted pose

dataset_video = YcbVideoDataset(video_path, mode, object_label, sample_interval, sample_num, video_num)

print("ycb_video has {0} entries".format(dataset_video.getLen()))

meta_index = 1
video_len = dataset_video.getLen()
start_index = 0
start_index_increment = 100
averaged_quaternion_dict = {}
total_distance_dict = {}

# Sample sample_num frames with sample_interval apart, starting at start_index
while start_index < video_len:
    sample_camera_poses = dataset_video.getCameraTransforms(start_index)
    initial_camera_pose = sample_camera_poses[0]

    # Store the camera transform from the initial camera pose to the sampled camera poses
    camera_transform_list = []
    for current in range(1, len(sample_camera_poses)):
        current_camera_pose = sample_camera_poses[current]
        camera_transform = computeCameraTransform(initial_camera_pose, current_camera_pose)
        camera_transform_list.append(camera_transform)


    # Estimate the pose for the sampled frames with DenseFusion with refinement
    df_estimator = DenseFusionEstimator(num_points, num_obj, model_checkpoint)
    outputs_list = dataset_video.getItem(start_index)
    
    distance = []
    predicted_poses = []
    ground_truth_quaternions = []

    for i in range(0, len(outputs_list)):
        points, choose, img, target, model_points, idx, quat = outputs_list[i]
        img = img.unsqueeze(0)
        points = points.unsqueeze(0)
        choose = choose.unsqueeze(0)
        idx = idx.unsqueeze(0)

        idx = idx - 1
        ground_truth_quaternions.append(quat)

        quat = quat.unsqueeze(0)
        
        points, choose, img, target, model_points, idx = Variable(points).cuda(), Variable(choose).cuda(), Variable(img).cuda(), Variable(target).cuda(), Variable(model_points).cuda(), Variable(idx).cuda()
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_q = pred_r[0,torch.argmax(pred_c)][[1,2,3,0]]
        pred_q /= pred_q.norm()
        distance.append(to_np(tensorAngularDiff(pred_q, quat.cuda()))*180/np.pi)
        
        predicted_poses.append(pred_q)

    #print("Precited_poses: {0}".format(predicted_poses))
    #print("Distance (degrees): {0}".format(distance))
    
    # Transform every non-first sampled frame to the first sampled frame
    predicted_initial_pose = predicted_poses[0]
    predicted_initial_pose_data = predicted_poses[0].cpu().detach().numpy()
    transformed_pose_list = applyTransform(predicted_poses[1:], camera_transform_list)
    transformed_pose_list.insert(0, predicted_initial_pose_data)

    averaged_pose = projectedAverageQuaternion(transformed_pose_list)
    #print("Averaged quaternion for index {0}: {1}".format(start_index, averaged_pose))
    
    difference = to_np(tensorAngularDiff(ground_truth_quaternions[0].unsqueeze(0), torch.from_numpy(averaged_pose).float()))*180/np.pi
    #print("Ground truth quaternion for index {0}: {1}".format(start_index, ground_truth_quaternions[0]))
    
    #print("Difference (degrees) between the predicted initial pose and ground truth initial pose: {0}".format(difference))
    
    averaged_quaternion_dict[start_index] = averaged_pose
    total_distance_dict[start_index] = difference
    start_index += start_index_increment
    
print("Averaged_quaternion list: {0}". format(averaged_quaternion_dict))
print("Distance (degrees) list: {0}".format(total_distance_dict))
