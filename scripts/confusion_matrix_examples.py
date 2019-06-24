from object_pose_utils.utils.confusion_matrix_builder import PoseConfusionMatrix
import torch
import numpy as np
from dense_fusion.network import PoseNet
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.image_processing import ImageNormalizer

dataset_root = '/home/mengyunx/DenseFusion/datasets/ycb'
mode = 'train'
object_list = [1]
model_checkpoint = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'
output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
                 otypes.IMAGE_CROPPED,
                 otypes.MODEL_POINTS_TRANSFORMED,
                 otypes.MODEL_POINTS,
                 otypes.OBJECT_LABEL,
                 otypes.QUATERNION]

num_objects = 21 #number of object classes in the dataset
num_points = 1000 #number of points on the input pointcloud

dataset = YCBDataset(dataset_root, mode=mode,
                     object_list = object_list,
                     output_data = output_format,
                     resample_on_error = True,
                     #preprocessors = [YCBOcclusionAugmentor(dataset_root), ColorJitter()],
                     #postprocessors = [ImageNormalizer(), PointShifter()],
                     postprocessors = [ImageNormalizer()],
                     image_size = [640, 480], num_points=1000)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.load_state_dict(torch.load(model_checkpoint))
estimator.cuda()

bins = np.load('/home/mengyunx/object_pose_utils/data/vertices.npy')

confusion_matrix = PoseConfusionMatrix(bins=bins, dataloader=dataloader, estimator=estimator)

#np.save('confusion_matrix.npy', confusion_matrix)

print(confusion_matrix.confusion_matrix)
pose = [-0.30901699, -0.80901699, -0.5       ,  0.        ]
pose_tensor = torch.tensor(pose)
con_prob = confusion_matrix.get_conditional_probability(pose_tensor)
print(con_prob)
