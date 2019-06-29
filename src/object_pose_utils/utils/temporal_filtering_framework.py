from object_pose_utils.utils.multi_view_utils import get_prediction, computeCameraTransform, applyTransform
import numpy as np

class TemporalFilteringFramework(object):
    # Update function compares prediction to measurement and updates the new pose
    def __init__(self, dataloader, estimator, update_function):
        self.frames_data = []
        self.frames_gt = []

        # Since a video dataloader is passed, the temporal filtering framework only happens within the subvideo sampled starting from the first index specified
        video = dataloader.dataset.getItem(0)
        self.camera_transforms = dataloader.dataset.getCameraTransforms(0)
  
        for i in range(0, len(video)):
            data = video[i]
            points, choose, img, target, model_points, idx, quat = data
            new_data = (points.unsqueeze(0),
                        choose.unsqueeze(0),
                        img.unsqueeze(0),
                        target.unsqueeze(0),
                        model_points.unsqueeze(0),
                        idx.unsqueeze(0),
                        quat.unsqueeze(0))
            # for item in new_data:        
            self.frames_data.append(new_data)
            self.frames_gt.append(quat)

        self.estimator = estimator
        self.current_frame_num = 0
        self.current_rotation, _ = get_prediction(estimator, self.frames_data[0])
        self.current_rotation = self.current_rotation.cpu().detach().numpy()
        self.update_function = update_function

    # The predict step where the new pose is propogated from motion model
    def predict(self, frame_to_predict):
        # Camera transform from the global frame to current
        camera_transform_current = self.camera_transforms[frame_to_predict-1]
        # Camera transform from the global frame to next
        camera_transform_next = self.camera_transforms[frame_to_predict]

        # Camera transform from current to next
        relative_camera_transform = computeCameraTransform(np.array(camera_transform_current), np.array(camera_transform_next))

        predicted_pose = applyTransform([self.current_rotation], [relative_camera_transform])
        predicted_pose = predicted_pose[0]

        return predicted_pose

    # The measurement step where the new pose is taken from measurement 
    def measure(self, frame_to_measure):
        predicted_pose, _ = get_prediction(self.estimator, self.frames_data[frame_to_measure])
        predicted_pose = predicted_pose.cpu().detach().numpy()
        return predicted_pose
        
    def propogate(self):
        if self.current_frame_num+1 < len(self.frames_data):
            # Prediction step
            # Propogate with motion prior
            new_rot_from_motion = self.predict(self.current_frame_num+1) 
            
            # Measurement step
            new_rot_from_measure = self.measure(self.current_frame_num+1)

            # Update step
            updated_state_estimate = self.update_function(new_rot_from_motion, new_rot_from_measure)
            updated_state_estimate = updated_state_estimate.reshape(4)
            updated_state_estimate /= np.linalg.norm(updated_state_estimate)
            self.current_rotation = updated_state_estimate
            self.current_frame_num += 1
            return 0
        else:
            return 1

    def get_frame_number(self):
        return self.current_frame_num
        
