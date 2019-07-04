from object_pose_utils.utils.confusion_matrix_builder import ConfMatrixEstimator
import torch
import numpy as np
from dense_fusion.network import PoseNet
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset as YCBVideoDataset
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.image_processing import ImageNormalizer
from object_pose_utils.utils.multi_view_utils import get_prediction
from object_pose_utils.utils.pose_processing import meanShift,tensorAngularDiff
from generic_pose.utils import to_np
import os.path


def evaluate_estimator(estimator, obj_id, interval, video_len):
    dataset_root = '/home/mengyunx/DenseFusion/datasets/ycb/YCB_Video_Dataset/'
    mode = 'test'
    object_list = [obj_id]
    model_checkpoint = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'
    output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
                     otypes.IMAGE_CROPPED,
                     otypes.MODEL_POINTS_TRANSFORMED,
                     otypes.MODEL_POINTS,
                     otypes.OBJECT_LABEL,
                     otypes.QUATERNION]

    num_objects = 21
    num_points = 1000

    ycb_dataset = YCBDataset(dataset_root, mode=mode,
                             object_list = object_list,
                             output_data = output_format,
                             resample_on_error = True,
                             #preprocessors = [YCBOcclusionAugmentor(dataset_root), ColorJitter()],
                             #postprocessors = [ImageNormalizer(), PointShifter()],
                             postprocessors = [ImageNormalizer()],
                             image_size = [640, 480], num_points=1000)

    ycb_video_dataset = YCBVideoDataset(ycb_dataset, interval, video_len)
    
    ycb_video_dataset.setObjectId(obj_id)
    eval_result = {}
    for v_id in ycb_video_dataset.getVideoIds():
        ycb_video_dataset.setVideoId(v_id)
        dataloader = torch.utils.data.DataLoader(ycb_video_dataset, batch_size=1, shuffle=False, num_workers=20)
        angular_error_list = []
        log_likelihood_list = []
        for i, (data, trans) in enumerate(dataloader, 0):
            # In each video, for each subvideo, perform the evaluation
            points, choose, img, target, model_points, idx, quat = data[0]
            q_gt = quat[0]
            #import IPython; IPython.embed()
            estimator.fit(data, trans)
            lik = estimator.likelihood(q_gt)
            q_est = estimator.mode()

            # Calculate angular error
            angular_error_degrees = to_np(tensorAngularDiff(torch.tensor(q_est).float().cuda(), quat.cuda()))*180/np.pi
            # Calculate log likelihood
            log_likelihood = np.log(lik)
            #import IPython; IPython.embed()
            angular_error_list.append(angular_error_degrees)
            log_likelihood_list.append(log_likelihood)
            if i % 100 == 0:
                print("Subvideo {0} has been evaluated".format(i))
        eval_result[v_id] = (angular_error_list, log_likelihood_list)
        print("Video {0} is evaluated".format(v_id))
    return eval_result

# ----- Evaluation code on Confusion Matrix -----
obj_list = list(range(1,22))
interval = 20
video_len = 3

my_path = os.path.abspath(os.path.dirname(__file__))
precomputed_path_prefix = os.path.join(my_path, "..", "precomputed")
for obj_id in obj_list:
    conf_matrix = np.load(os.path.join(precomputed_path_prefix, "confusion_matrices", "{0}_confusion_matrix.npy".format(str(obj_id))))
    conf_matrix_estimator = ConfMatrixEstimator(conf_matrix)
    conf_matrix_eval_result = evaluate_estimator(conf_matrix_estimator, obj_id, interval, video_len)

    print("----- Evaluation result of the confusion matrix method -----")
    for video_id, result in conf_matrix_eval_result.items():
        angular_error_list, log_likelihood_list = result
        print("Obj: {0}, Video: {1}, Avg. Angular Error (deg): {2}, Ave. Log Likelihood: {3}".format(str(obj_id), video_id, str(np.average(angular_error_list)), str(np.average(log_likelihood_list))))




# network_path_prefix = os.path.join(my_path, "..", "..", "networks")
# dataset_path_prefix = os.path.join(my_path, "..", "..", "datasets")

# dataset_root = os.path.join(dataset_path_prefix, "DenseFusion/datasets/ycb")estimator.load_state_dict(torch.load(model_checkpoint))
# #dataset_root = '/home/mengyunx/DenseFusion/datasets/ycb'

# mode = 'train'
# object_list = [1]

# model_checkpoint = os.path.join(dataset_path_prefix, "DenseFusion/trained_checkpoints/ycb/DenseFusion/trained_checpoints/ycb/pose_model_26_0.012863246640872631.pth")
# #model_checkpoint = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'

# output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
#                  otypes.IMAGE_CROPPED,
#                  otypes.MODEL_POINTS_TRANSFORMED,
#                  otypes.MODEL_POINTS,
#                  otypes.OBJECT_LABEL,
#                  otypes.QUATERNION]

# num_objects = 21 #number of object classes in the dataset
# num_points = 1000 #number of points on the input pointcloud

# dataset = YCBDataset(dataset_root, mode=mode,
#                      object_list = object_list,
#                      output_data = output_format,
#                      resample_on_error = True,
#                      #preprocessors = [YCBOcclusionAugmentor(dataset_root), ColorJitter()],
#                      #postprocessors = [ImageNormalizer(), PointShifter()],
#                      postprocessors = [ImageNormalizer()],
#                      image_size = [640, 480], num_points=1000)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# estimator = PoseNet(num_points = num_points, num_obj = num_objects)
# estimator.load_state_dict(torch.load(model_checkpoint))
# estimator.cuda()

# conf_matrix_estimator = ConfMatrixEstimator()

# #np.save('confusion_matrix.npy', confusion_matrix)

# pose = [-0.30901699, -0.80901699, -0.5       ,  0.        ]
# pose_tensor = torch.tensor(pose)
# con_prob = confusion_matrix.get_conditional_probability(pose_tensor)
# print(con_prob)

# # Evaluate the performance
# print("-------- Start the performance evaluation --------")
# mode = "valid"

# dataset_eval = YCBDataset(dataset_root, mode=mode,
#                      object_list = object_list,
#                      output_data = output_format,
#                      resample_on_error = True,
#                      #preprocessors = [YCBOcclusionAugmentor(dataset_root), ColorJitter()],
#                      #postprocessors = [ImageNormalizer(), PointShifter()],
#                      postprocessors = [ImageNormalizer()],
#                      image_size = [640, 480], num_points=1000)

# dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=1, shuffle=False, num_workers=0)

# count_matrix_better = 0
# count_densefusion_better = 0

# for i, data in enumerate(dataloader_eval, 0):
#     # Predicted pose and ground truth pose fetched from DenseFusion's estimator
#     predicted_pose, gt_pose = get_prediction(estimator, data)

#     # With the predicted pose, use the confusion matrix's distribution to find the most likely pose
#     conditional_prob = torch.tensor(confusion_matrix.get_conditional_probability(predicted_pose)).unsqueeze(1)

#     #import IPython; IPython.embed()
#     # Find the mode
#     #max_idx = torch.argmax(conditional_prob)
#     # Could also use multiple particles
#     num_modes = 3
#     max_idx = np.argsort(-1*conditional_prob.numpy(), axis=0)[:num_modes]
#     grid_vertices = torch.tensor(confusion_matrix.bins)
#     import IPython; IPython.embed()
#     starting_particles = grid_vertices[max_idx].unsqueeze(0)
#     mode_quats = meanShift(starting_particles.float(), grid_vertices.float(), conditional_prob.float())

#     distance = to_np(tensorAngularDiff(torch.tensor(mode_quats).float().cuda(), gt_pose.unsqueeze(0).cuda()))*180/np.pi
#     distance_densefusion = to_np(tensorAngularDiff(torch.tensor(predicted_pose).float().cuda(), gt_pose.unsqueeze(0).cuda()))*180/np.pi

#     if i % 100 == 0:
#         print("Validation image {0} / {1} has been evaluated".format(i, dataloader_eval.__len__()))
#     #print("Distance via confusion matrix(degrees): {0}".format(distance))
#     #print("Distance via densefusion (degrees): {0}".format(distance_densefusion))

#     if distance < distance_densefusion:
#         count_matrix_better += 1
#     elif distance > distance_densefusion:
#         count_densefusion_better += 1


# print("Confusion matrix method performs better {0} times, DenseFusion method performs better {1} times".format(count_matrix_better, count_densefusion_better))
    
    
