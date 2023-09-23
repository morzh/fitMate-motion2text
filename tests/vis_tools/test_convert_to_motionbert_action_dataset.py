import unittest
import numpy as np
from dataset_tools.convert_to_motionbert_action_dataset import DatasetFromVideo


class TestDatasetFromVideo(unittest.TestCase):
    """
    Test the following cases
    1.      NUMBER FRAMES < sample_frames_number
    1.a.    number samples < minimum_number_frames
    1.b.    minimum_number_frames <= number samples < sample_frames_number

    2.      NUMBER FRAMES == sample_frames_number

    3.      NUMBER FRAMES > sample_frames_number
    3.a.    sample_frames_number < number samples <  sample_frames_number + 2*number_cutoff_frames
    3.b.    sample_frames_number + 2*number_cutoff_frames <= number samples
    """
    def setUp(self):
        self.number_cutoff_frames = 30
        self.minimum_number_frames = 90
        self.sample_frames_number = 243
        self.number_joints = 17
        self.dataset_from_video = DatasetFromVideo(number_cutoff_frames=self.number_cutoff_frames,
                                                   minimum_number_frames=self.minimum_number_frames,
                                                   sample_frames_number=self.sample_frames_number,
                                                   number_joints=self.number_joints)
        self.number_tests = 1500

    def test_case_1a(self):
        """
            Case 1.a.
            number samples < minimum_number_frames
        Returns:
            should return empty numpy array, shape[0] = 0
        """
        for index in range(self.number_tests):
            sample_1a_in = np.random.random((np.random.randint(0, self.minimum_number_frames), self.number_joints, 3))
            sample_1a_out = self.dataset_from_video.get_samples(sample_1a_in)
            self.assertEqual(sample_1a_out.shape[0], 0)

    def test_case_1b(self):
        """
            Case 1.b.
            self.minimum_number_frames <= number_frames < sample_frames_number
        Returns:

        """
        for index in range(self.number_tests):
            sample_1b_in = np.random.random((np.random.randint(self.minimum_number_frames, self.sample_frames_number), self.number_joints, 3))
            sample_1b_out = self.dataset_from_video.get_samples(sample_1b_in)

            number_frames = sample_1b_in.shape[0]
            number_additional_frames = self.sample_frames_number - number_frames
            number_pre_additional_frames = int(np.ceil(0.5 * number_additional_frames))
            number_post_additional_frames = int(np.floor(0.5 * number_additional_frames))

            self.assertEqual(sample_1b_out.shape[0], 2)
            self.assertEqual(sample_1b_out.shape[1], self.sample_frames_number)
            self.assertTrue((sample_1b_out[0][:number_pre_additional_frames] == sample_1b_in[0]).all())
            self.assertTrue((sample_1b_out[0][-number_post_additional_frames-1:] == sample_1b_in[-1]).all())

    def test_case_2(self):
        """
            Case 2.
            number_frames == sample_frames_number
        Returns:

        """
        for index in range(self.number_tests):
            sample_2_in = np.random.random((self.sample_frames_number, self.number_joints, 3))
            sample_2_out = self.dataset_from_video.get_samples(sample_2_in)

            self.assertEqual(sample_2_out.shape[0], 2)
            self.assertEqual(sample_2_out.shape[1], self.sample_frames_number)

    def test_case_3a(self):
        """
            Case 3.a.
            sample_frames_number < number samples < sample_frames_number + 2 * number_cutoff_frames
        Returns:

        """
        for index in range(self.number_tests):
            sample_3a_in = np.random.random((np.random.randint(self.sample_frames_number + 1, self.sample_frames_number + 2 * self.number_cutoff_frames - 1), self.number_joints, 3))
            sample_3a_out = self.dataset_from_video.get_samples(sample_3a_in)

            number_frames = sample_3a_in.shape[0]
            number_additional_frames = number_frames - self.sample_frames_number
            number_pre_additional_frames = int(np.ceil(0.5 * number_additional_frames))
            number_post_additional_frames = int(np.floor(0.5 * number_additional_frames))

            if number_post_additional_frames == 0:
                sample_check = sample_3a_in[number_pre_additional_frames:]
            else:
                sample_check = sample_3a_in[number_pre_additional_frames: -number_post_additional_frames]

            self.assertEqual(sample_3a_out.shape[0], 2)
            self.assertEqual(sample_3a_out.shape[1], self.sample_frames_number)
            self.assertTrue((sample_check == sample_3a_out[0]).all())

    def test_case_3b(self):
        """
            Case 3.b.
            sample_frames_number + 2*number_cutoff_frames < number samples
        Returns:
        """
        for index in range(self.number_tests):
            sample_3b_in = np.random.random((np.random.randint(self.sample_frames_number + 2 * self.number_cutoff_frames, int(1e4)), self.number_joints, 3))
            sample_3b_out = self.dataset_from_video.get_samples(sample_3b_in)

            number_frames = sample_3b_in.shape[0]
            number_samples = (number_frames - 2 * self.number_cutoff_frames) // self.sample_frames_number
            samples = sample_3b_in[self.number_cutoff_frames: self.number_cutoff_frames + number_samples * self.sample_frames_number]

            for index_sample in range(number_samples - 1):
                self.assertTrue((sample_3b_out[index_sample][0] == samples[index_sample * self.sample_frames_number:  (index_sample + 1) * self.sample_frames_number]).all())


