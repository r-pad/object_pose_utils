from object_pose_utils.datasets.linemod_dataset import LinemodDataset
from object_pose_utils.datasets.ycb_dataset import YcbDataset
from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset
import torch

#dataset_linemod = LinemodDataset("/home/mengyunx/DenseFusion/datasets/linemod/Linemod_preprocessed",
#                                    "test",
#                                    [1,2,4,5,6,8,9,10,11,12,13,14,15])
#dataloader_linemod = torch.utils.data.DataLoader(dataset_linemod, batch_size=1, shuffle=False, num_workers=10)

#for i, data in enumerate(dataloader_linemod, 0):
#    print(i)
#    print(data)
    
#print("linemod has {0} entries".format(dataloader_linemod.__len__()))

#dataset_ycb = YcbDataset("/home/mengyunx/DenseFusion/datasets/ycb",
#                          "train",
#                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])

#dataloader_ycb = torch.utils.data.DataLoader(dataset_ycb, batch_size=1, shuffle=False, num_workers=20)
#print("ycb has {0} entries".format(dataset_ycb.__len__()))
#for i, data in enumerate(dataloader_ycb, 0):
#    print(i)
#    print(data)

dataset_video = YcbVideoDataset("/home/mengyunx/DenseFusion/datasets/ycb", "train", [1], 5, 1)

dataloader_video = torch.utils.data.DataLoader(dataset_video, batch_size=1, shuffle=False, num_workers=0)

print("ycb_video has {0} entries".format(dataset_video.__len__()))

for i, data in enumerate(dataloader_video, 0):
    print(i)
    print(data)
