import numpy as np
from quat_math import quaternion_from_matrix, quaternion_multiply

# Apply each camera transform to each pose
def applyTransform(poses, cameraTransforms):
    # Input: poses: non-first frame poses (quaternion)
    #        cameraTransforms: the camera transforms from the first frame to the ith frame 3x4
    assert len(poses) == len(cameraTransforms), "inputs don't have the same length"
    transformed = []
    for i in range(0, len(poses)):
        current_pose = poses[i]
        camera_transform = cameraTransforms[i]
        R1c = camera_transform[:, :3].transpose()
        R1c_padded = np.identity(4)
        R1c_padded[:3, :3] = R1c
        R1c_quat = quaternion_from_matrix(R1c_padded)
        first_frame_pose = quaternion_multiply(R1c_quat, current_pose.cpu().detach().numpy())
        transformed.append(first_frame_pose)

    return transformed

# Compute the camera transform matrix from the initial frame to the current frame
def computeCameraTransform(initial_camera_matrix, current_camera_matrix):
    # Initial camera matrix = from world to initial frame (Tiw) 3x4
    # current camera matrix = from world to current frame (Tcw) 3x4
    Rcw = current_camera_matrix[:, :3]
    tcw = current_camera_matrix[:, 3]
    Riw = initial_camera_matrix[:, :3]
    tiw = initial_camera_matrix[:, 3]

    Rci = np.matmul(Rcw, Riw.transpose())
    tci = tcw - np.matmul(Rci, tiw)

    Tci = np.zeros((3,4))
    Tci[:, :3] = Rci
    Tci[:, 3] = tci

    return Tci
                                                                        
