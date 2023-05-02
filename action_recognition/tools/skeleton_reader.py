import cv2
import mediapipe as mp
import numpy as np

from settings import mediapipe_options

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class SkeletonReader:
    def __init__(self):
        self.mp_pose = mp_pose.Pose(**mediapipe_options)
        self.last_image = None
        self.last_mp_points = None

    def get_skeleton(self, image):
        self.last_image = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(image)
        self.last_mp_points = results
        if results.pose_landmarks is not None:
            np_results = np.array([(joint.x, joint.y, joint.z) for joint in
                                   list(results.pose_landmarks.landmark)])
        else:
            np_results = None
        return np_results

    def draw_pose_points(self, image=None, points=None):
        if image is None:
            image = self.last_image
        if points is None:
            points = self.last_mp_points

        mp_drawing.draw_landmarks(
            image,
            points.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        return image


