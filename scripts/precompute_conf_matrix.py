from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset as YCBVideoDataset
from object_pose_utils.datasets.image_processing import ImageNormalizer
from object_pose_utils.utils.pose_processing import quatAngularDiffBatch
import os.path
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
import torch
from dense_fusion.network import PoseNet
import numpy as np
from torch.autograd import Variable
from generic_pose.utils import to_np, to_var

# Map the pose to the closest bin
def get_index(bins, pose):
    dists = quatAngularDiffBatch(to_np(pose), bins)
    bin_id = np.argmin(dists)
    return bin_id
                    
my_path = os.path.abspath(os.path.dirname(__file__))
network_path_prefix = os.path.join(my_path, "..", "networks")
dataset_path_prefix = os.path.join(my_path, "..", "datasets")
dataset_root = os.path.join(dataset_path_prefix, "DenseFusion/datasets/ycb")

# YCB dataset has 21 objects that have object id 1-21.
object_list = list(range(1,22))
#dataset_root = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'
mode = "valid"

output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
                 otypes.IMAGE_CROPPED,
                 otypes.MODEL_POINTS_TRANSFORMED,
                 otypes.MODEL_POINTS,
                 otypes.OBJECT_LABEL,
                 otypes.QUATERNION]

model_checkpoint = os.path.join(dataset_path_prefix, "DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth")

num_objects = 21
num_points = 1000

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.load_state_dict(torch.load(model_checkpoint))
estimator.cuda()

bins = np.load('/home/mengyunx/object_pose_utils/precomputed/vertices.npy')

for object_id in object_list:
    confusion_matrix = np.zeros((len(bins), len(bins)))
    ycb_dataset = YCBDataset(dataset_root, mode=mode,
                             object_list = [object_id],
                             output_data = output_format,
                             postprocessors = [ImageNormalizer()],
                             image_size = [640, 480], num_points=1000)

    dataloader = torch.utils.data.DataLoader(ycb_dataset, batch_size=1, shuffle=False, num_workers=20)

    save_directory = "/home/mengyunx/object_pose_utils/precomputed/confusion_matrices/{0}_confusion_matrix.npy".format(str(object_id))

    if not os.path.exists(save_directory):
        
        for i, data in enumerate(dataloader, 0):
            points, choose, img, target, model_points, idx, quat = data
            idx = idx - 1
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            pred_q = pred_r[0,torch.argmax(pred_c)][[1,2,3,0]]
            pred_q /= pred_q.norm()

            col = get_index(bins, quat[0])
            row = get_index(bins, pred_q)
            confusion_matrix[row, col] += 1
            if i % 100 == 0:
                print("Image {0} / {1} has been processed".format(i, dataloader.__len__()))


        np.save(save_directory, confusion_matrix)
        print("Confusion matrix for object {0} has been precomputed".format(str(object_id)))

    else:
        print("Confusion matrix for object {0} has not been precomputed because already exists".format(str(object_id)))
