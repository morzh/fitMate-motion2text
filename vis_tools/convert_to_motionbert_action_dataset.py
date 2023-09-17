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


class DatasetClampDivideStrategy(DatasetStrategy):
    def __init__(self, soft_cutoff_frames: int,  minimum_number_frames=50, sample_frames_length=243, number_joints=17):
        super().__init__()
        self.soft_cutoff_frames_ = soft_cutoff_frames
        self.minimum_number_frames_ = minimum_number_frames
        self.sample_frames_length_ = sample_frames_length
        self.number_joints_ = number_joints
        self.second_person_fake_ = np.zeros((1, self.sample_frames_length_, self.number_joints_, 3))

    def get_samples(self, data: np.ndarray) -> np.ndarray:
        """

        Args:
            data:

        Returns:
            numpy array in [number_samples, 2, number_frames, number_joints, 3] format.
            It assumes two persons are interacting (second person could be faked)
        """
        number_frames = data.shape[0]
        if number_frames > self.sample_frames_length_:
            if (number_frames - 2 * self.soft_cutoff_frames_) < self.sample_frames_length_:
                number_new_frames = number_frames - self.sample_frames_length_
                number_frames_pre_cut = np.ceil(number_new_frames)
                number_frames_post_cut = np.floor(number_new_frames)
                sample = data[number_frames_pre_cut: number_frames_post_cut]
                return self.fake_second_person_(sample)
            elif (number_frames - 2 * self.soft_cutoff_frames_) == self.sample_frames_length_:
                sample = data[self.soft_cutoff_frames_, -self.soft_cutoff_frames_]
                return self.fake_second_person_(sample)
            else:
                number_samples = (number_frames - 2 * self.soft_cutoff_frames_) % self.sample_frames_length_
                samples = np.zeros((number_samples, 2, self.sample_frames_length_, self.number_joints_, 3))
                for sample_index in range(number_samples):
                    index_first = self.soft_cutoff_frames_ + sample_index * self.sample_frames_length_
                    index_second = self.soft_cutoff_frames_ + (sample_index + 1) * self.sample_frames_length_
                    sample = data[index_first: index_second]
                    samples[sample_index] = self.fake_second_person_(sample)
                return samples
        elif number_frames == self.sample_frames_length_:
            return self.fake_second_person_(data)
        else:
            if number_frames < self.minimum_number_frames_:
                return np.empty((0, 2, 243, 17, 3))
            number_new_frames = number_frames - self.sample_frames_length_
            number_frames_pre_add = np.ceil(number_new_frames)
            number_frames_post_add = np.floor(number_new_frames)
            sample_pre_add = np.repeat(data[0], number_frames_pre_add)
            sample_post_add = np.repeat(data[-1], number_frames_post_add)
            sample = np.vstack((sample_pre_add, data, sample_post_add))
            return self.fake_second_person_(sample)

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

    dataset_generator = DatasetClampDivideStrategy(60)

    for file in files:
        json_filepath = os.path.join(folder_annotations, file)
        with open(json_filepath, 'r') as f:
            annotation = json.load(f)

        poses = [p['pose'] for p in annotation['annotations_PCT']]
        poses = np.array(poses)
        dataset_generator.get_samples(poses)



