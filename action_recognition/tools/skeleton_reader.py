import cv2
import mediapipe as mp
import numpy as np

from action_recognition.settings import mediapipe_options

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class SkeletonReader:
    def __init__(self):
        self.mp_pose = mp_pose.Pose(**mediapipe_options)
        self.last_image = None
        self.last_mp_points = None
        self._img_boarder = (0, 0, 0, 0)  # top, bottom, left, right

    def get_skeleton(self, image):
        self.last_image = image
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.litterbox_on(image)
        results = self.mp_pose.process(image)
        self.last_mp_points = results
        if results.pose_landmarks is not None:
            np_results = np.array([(joint.x, joint.y, joint.z) for joint in
                                   list(results.pose_landmarks.landmark)])
        else:
            np_results = None
        return np_results

    def litterbox_on(self, image):
        """ Make image aspect ratio equal 1 """
        h, w, _ = image.shape
        if h > w:
            left, right = divmod(h - w, 2)
            right += left
            self._img_boarder = (0, 0, left, right)
        elif w > h:
            top, bottom = divmod(w - h, 2)
            bottom += top
            self._img_boarder = (top, bottom, 0, 0)
        else:
            self._img_boarder = (0, 0, 0, 0)
        image = cv2.copyMakeBorder(
            image, *self._img_boarder, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return image

    def litterbox_off(self, image):
        top, bottom, left, right = self._img_boarder
        if top != 0:
            image = image[bottom: -top, ...]
        if left != 0:
            image = image[:, left: -right, :]
        return image

    def draw_pose_points(self, image=None, points=None, littrebox=True):
        if image is None:
            image = self.last_image
        if littrebox:
            image = self.litterbox_on(image)

        if points is None:
            points = self.last_mp_points

        mp_drawing.draw_landmarks(
            image,
            points.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if littrebox:
            image = self.litterbox_off(image)
        # if to_rgb:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
