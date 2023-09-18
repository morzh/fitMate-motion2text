import json
import os
from abc import abstractmethod
import numpy as np


class DatasetStrategy(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_samples(self, data: np.ndarray) -> np.ndarray:
        pass


class DatasetFromVideo(DatasetStrategy):
    def __init__(self, number_cutoff_frames=30, minimum_number_frames=50, sample_frames_number=243, number_joints=17):
        """

        Args:
            number_cutoff_frames:
            minimum_number_frames:
            sample_frames_number:
            number_joints:
        """
        super().__init__()
        self.soft_cutoff_frames_ = number_cutoff_frames
        self.minimum_number_frames_ = minimum_number_frames
        self.sample_frames_number_ = sample_frames_number
        self.number_joints_ = number_joints
        self.second_person_fake_ = np.zeros((1, self.sample_frames_number_, self.number_joints_, 3))

    def get_samples(self, data: np.ndarray) -> np.ndarray:
        """
        The following cases are considered:
            1.      NUMBER FRAMES < self.sample_frames_length_
            1.a.    number_samples < self.minimum_number_frames
            1.b.    self.minimum_number_frames <= number_samples < self.sample_frames_length_

            2.      NUMBER FRAMES == self.sample_frames_length_

            3.      NUMBER FRAMES > self.sample_frames_length_
            3.a.    self.sample_frames_length_ < number_frames <  self.sample_frames_length_ + 2*self.number_cutoff_frames
            3.b.    self.sample_frames_length_ + 2*number_cutoff_frames <= number_frames

        Args:
            data: numpy array in [frames, joints, 3]  format

        Returns:
            numpy array in [number_samples, 2, number_frames, number_joints, 3] format, number_samples >= 0.
            It assumes two persons are interacting (second person is a faked person)
        """
        number_frames = data.shape[0]

        if number_frames < self.minimum_number_frames_:  # case 1.a
            return np.empty((0, 2, 243, 17, 3))
        elif self.minimum_number_frames_ <= number_frames < self.sample_frames_number_:  # case 1.b
            number_additional_frames = self.sample_frames_number_ - number_frames
            number_pre_additional_frames = int(np.ceil(0.5 * number_additional_frames))
            number_post_additional_frames = int(np.floor(0.5 * number_additional_frames))
            sample_pre_add = np.repeat(np.expand_dims(data[0], axis=0), number_pre_additional_frames, axis=0)
            sample_post_add = np.repeat(np.expand_dims(data[-1], axis=0), number_post_additional_frames, axis=0)
            sample = np.vstack((sample_pre_add, data, sample_post_add))
            return self.fake_second_person_(sample)
        elif number_frames == self.sample_frames_number_:  # case 2.
            return self.fake_second_person_(data)
        elif self.sample_frames_number_ < number_frames < self.sample_frames_number_ + 2 * self.soft_cutoff_frames_:  # case 3.a
            number_additional_frames = number_frames - self.sample_frames_number_
            number_pre_additional_frames = int(np.ceil(0.5 * number_additional_frames))
            number_post_additional_frames = int(np.floor(0.5 * number_additional_frames))
            if number_post_additional_frames == 0:
                sample = data[number_pre_additional_frames:]
            else:
                sample = data[number_pre_additional_frames: -number_post_additional_frames]
            return self.fake_second_person_(sample)
        else:  # case 3.b
            number_samples = (number_frames - 2 * self.soft_cutoff_frames_) // self.sample_frames_number_
            samples = data[self.soft_cutoff_frames_: self.soft_cutoff_frames_ + number_samples * self.sample_frames_number_]
            samples = samples.reshape(number_samples, self.sample_frames_number_, self.number_joints_, 3, order='C')
            samples = np.expand_dims(samples, axis=1)
            zeros = np.zeros(samples.shape)
            samples = np.hstack((samples, zeros))
            '''
            samples = np.zeros((number_samples, 2, self.sample_frames_number_, self.number_joints_, 3))
            for sample_index in range(number_samples):
                index_first = self.soft_cutoff_frames_ + sample_index * self.sample_frames_number_
                index_second = self.soft_cutoff_frames_ + (sample_index + 1) * self.sample_frames_number_
                sample = data[index_first: index_second]
                samples[sample_index] = self.fake_second_person_(sample)
            '''
            return samples

    def fake_second_person_(self,  data: np.ndarray) -> np.ndarray:
        sample = np.expand_dims(data, axis=0)
        sample = np.vstack((sample, self.second_person_fake_))
        return sample

    def normalize(self, data: np.ndarray, image_shape: np.ndarray) -> np.ndarray:
        pass


if '__name__' == '__main__':

    folder_annotations = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions/annotations'
    folder_dataset = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions/motionBERT_dataset'
    files = [f for f in os.listdir(folder_annotations) if os.path.isfile(os.path.join(folder_annotations, f)) and f.endswith('json')]

    dataset_generator = DatasetFromVideo()

    for file in files:
        json_filepath = os.path.join(folder_annotations, file)
        with open(json_filepath, 'r') as f:
            annotation = json.load(f)

        poses = [p['pose'] for p in annotation['annotations_PCT']]
        poses = np.array(poses)
        dataset_generator.get_samples(poses)



