import torch
import os
# Import video dataloader
from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset
from object_pose_utils.utils.multi_view_utils import applyTransform, computeCameraTransform
from quat_math.rotation_averaging import projectedAverageQuaternion

# Import estimator
from dense_fusion.evaluate import DenseFusionEstimator

# Choose the video
video_num = 1
video_path = '/home/mengyunx/DenseFusion/datasets/ycb'
mode = 'train'
object_label = 1
model_checkpoint = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'
num_points = 1000
num_obj = 21

# Choose a sample interval (sample every n frames. For n = 3, samples are [1] 2 3 [4] 5 6 for start frame = 0
sample_interval = 5
sample_num = 3

# For each sequence of samples, apply camera poses to get the final frame's predicted pose

dataset_video = YcbVideoDataset(video_path, mode, object_label, sample_interval, sample_num, video_num)

print("ycb_video has {0} entries".format(dataset_video.getLen()))

meta_index = 1
video_len = dataset_video.getLen()
start_index = 0
start_index_increment = 4
averaged_quaternion_dict = {}

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

    color_images = dataset_video.getImage(start_index)
    depth_images = dataset_video.getDepthImage(start_index)
    metadata = dataset_video.getMetaData(start_index, mask=True, bbox=True)
    predicted_poses = []
    for i in range(0, len(color_images)):
        img = color_images[i]
        depth = depth_images[i]
        mask = metadata[i]['mask']
        cmin, rmin, w, h = metadata[i]['bbox']
        cmax = cmin+w
        rmax = rmin+h
        bbox = (rmin, rmax, cmin, cmax)
        #import IPython; IPython.embed()
        #print("img:{0}".format(img))
        #print("depth: {0}".format(depth))
        #print("mask: {0}".format(mask))
        #print("bbox: {0}".format(bbox))
        #print("object_label: {0}".format(object_label))
        pred_r, pred_t = df_estimator(img, depth, mask, bbox, object_label)
        predicted_poses.append(pred_r)

    # Transform every non-first sampled frame to the first sampled frame

    print("Predicted_poses: {0}".format(predicted_poses[1:]))
    transformed_pose_list = applyTransform(predicted_poses[1:], camera_transform_list)
    transformed_pose_list.append(predicted_poses[0].cpu().detach().numpy())
    averaged_pose = projectedAverageQuaternion(transformed_pose_list)
    averaged_quaternion_dict[start_index] = averaged_pose
    start_index += start_index_increment

print("Averaged_quaternion list: {0}". format(averaged_quaternion_dict))
