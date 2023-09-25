from typing import Union
from abc import abstractmethod
import numpy as np


class DatasetBase(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_samples(self, data: np.ndarray) -> np.ndarray:
        pass


class DatasetFromVideoAnnotations(DatasetBase):
    def __init__(self, number_cutoff_frames: int = 30, minimum_number_frames: int = 50, sample_frames_number: int = 243, number_joints: int = 17):
        """

        Args:
            number_cutoff_frames:
            minimum_number_frames:
            sample_frames_number:
            number_joints:
        """
        super().__init__()
        self.soft_cutoff_frames = number_cutoff_frames
        self.minimum_number_frames = minimum_number_frames
        self.sample_frames_number = sample_frames_number
        self.number_joints = number_joints
        self.pct_multiplier_score = 0.11
        self.pct_multiplier_outside_image = 0.2
        self.pct_multiplier_sigma_score = 0.9

    def reassign_pct_joints_scores(self, poses: np.ndarray, image_shape: Union[np.ndarray, tuple]) -> np.ndarray:
        """
        1. Multiply score by self.pct_multiplier_score
        2. Multiply score of joints which are outside image frame by self.pct_multiplier_outside_image
        3. В предположении, что суставы не могут резко менять координаты, посчитать среднее смещение и
            разброс (sigma_x, sigma_y или среднеквадратическое покоординатно) по X и Y и штрафовать суставы,
            которые переместились больше, чем на sigma_x, sigma_y

        Args:
            poses: [number_frames, 17, 3] numpy array, where N -- number of frames
            image_shape: ndarray or tuple in [width, height] format
        Returns:
            numpy ndarray in [number_frames, number_joints, 3] format
        """
        assert poses.shape[-1] == 3

        width, height = image_shape

        poses[:, :, :, 2] *= self.pct_multiplier_score

        poses_x = poses[:, :, :, 0]
        poses_y = poses[:, :, :, 1]
        poses_scores = poses[:, :, :, 2]

        poses_outside_video_mask = np.logical_or(np.logical_or(poses_x < 0,  poses_x >= height), np.logical_or(poses_y < 0,  poses_x >= width))
        poses_scores[poses_outside_video_mask] *= self.pct_multiplier_outside_image

        poses_dt = poses[:, 1:, :, :2] - poses[:, :-1, :, :2]
        poses_dt_distances = np.linalg.norm(poses_dt, axis=3)
        mean_dt = np.mean(poses_dt_distances, axis=2)
        std_dt = np.std(poses_dt_distances, axis=2)
        poses_outside_sigma_mask = (poses_dt_distances - np.expand_dims(mean_dt, axis=2)) > np.expand_dims(std_dt, axis=2)
        poses_scores_partial = poses_scores[:, 1:]
        poses_scores_partial[poses_outside_sigma_mask] *= self.pct_multiplier_sigma_score

        return poses

    def get_samples(self, data: np.ndarray) -> np.ndarray:
        """
        The following cases are considered:
            1.      NUMBER FRAMES < self.sample_frames_number_
            1.a.    number_frames < self.minimum_number_frames ==> empty (0, 2, 243, 17, 3) array
            1.b.    self.minimum_number_frames <= number_frames < self.sample_frames_length_ ==>  (1, 2, 243, 17, 3) array with constant extrapolation

            2.      number_frames == self.sample_frames_number_

            3.      NUMBER FRAMES > self.sample_frames_number_
            3.a.    self.sample_frames_number_ < number_frames <  self.sample_frames_number_ + 2*self.number_cutoff_frames
            3.b.    self.sample_frames_number_ + 2*number_cutoff_frames <= number_frames

        Args:
            data: numpy ndarray in [frames, joints, 3]  format

        Returns:
            numpy array in [number_samples, 2, number_frames, number_joints, 3] format, number_samples >= 0.
            It assumes two persons are interacting (second person is a faked person)
        """
        number_frames = data.shape[0]

        if number_frames < self.minimum_number_frames:  # case 1.a
            return np.empty((0, 243, 17, 3))
        elif self.minimum_number_frames <= number_frames < self.sample_frames_number:  # case 1.b
            number_additional_frames = self.sample_frames_number - number_frames
            number_pre_additional_frames = int(np.ceil(0.5 * number_additional_frames))
            number_post_additional_frames = int(np.floor(0.5 * number_additional_frames))
            sample_pre_add = np.repeat(np.expand_dims(data[0], axis=0), number_pre_additional_frames, axis=0)
            sample_post_add = np.repeat(np.expand_dims(data[-1], axis=0), number_post_additional_frames, axis=0)
            sample = np.vstack((sample_pre_add, data, sample_post_add))
            return np.expand_dims(sample, axis=0)
        elif number_frames == self.sample_frames_number:  # case 2.
            return data
        elif self.sample_frames_number < number_frames < self.sample_frames_number + 2 * self.soft_cutoff_frames:  # case 3.a
            number_additional_frames = number_frames - self.sample_frames_number
            number_pre_additional_frames = int(np.ceil(0.5 * number_additional_frames))
            number_post_additional_frames = int(np.floor(0.5 * number_additional_frames))
            if number_post_additional_frames == 0:
                sample = data[number_pre_additional_frames:]
            else:
                sample = data[number_pre_additional_frames: -number_post_additional_frames]
            return np.expand_dims(sample, axis=0)
        else:  # case 3.b
            number_samples = (number_frames - 2 * self.soft_cutoff_frames) // self.sample_frames_number
            samples = data[self.soft_cutoff_frames: self.soft_cutoff_frames + number_samples * self.sample_frames_number]
            samples = samples.reshape(number_samples, self.sample_frames_number, self.number_joints, 3, order='C')
            return samples

    @staticmethod
    def normalize(data: np.ndarray, image_shape: Union[np.ndarray, tuple]) -> np.ndarray:
        """
        Normalize joint's coordinates using image dimensions
        Args:
            data: joints in [joints, 2] or [joints, 3] format
            image_shape: image dimensions [width, height]

        Returns:
            normalized (by image size) joints coordinates
        """
        if isinstance(image_shape, np.ndarray):
            assert image_shape.shape == 1
            assert image_shape.shape[0] == 2
            assert image_shape.dtype == np.int32
        elif isinstance(image_shape, tuple):
            assert len(image_shape) == 2

        width, height = image_shape
        if width >= height:
            data[:, :, :2] = data[:, :, :2] / width * 2 - 1
        else:
            data[:, :, :2] = data[:, :, :2] / height * 2 - 1

        return data
