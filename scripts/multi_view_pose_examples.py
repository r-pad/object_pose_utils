from object_pose_utils.utils.multi_view_pose import estimate_average_quaternion

# Choose video
video_num = 1 # the prefix 00xx/xxxx in the file id
dataset_root = '/home/mengyunx/DenseFusion/datasets/ycb'
mode = 'train'
object_label = 1 # 0 is background, 1 is 002/master_chef_can...
model_checkpoint = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'

# Choose a sample interval (sample every n frames. For n = 3, samples are [1] 2 3 [4] 5 6 for start frame = 0
sample_interval = 50
sample_num = 10
start_index = 0
start_index_increment = 100

averaged_quaternion_dict, total_distance_dict = estimate_average_quaternion(dataset_root, mode, model_checkpoint, video_num, sample_num, sample_interval, start_index, start_index_increment, object_label)

print("Averaged_quaternion list: {0}". format(averaged_quaternion_dict))
print("Distance (degrees) list: {0}".format(total_distance_dict)) 
