import numpy as np
from vis_tools.convert_to_motionbert_action_dataset import DatasetClampDivideStrategy

'''
Test the following cases
1.      NUMBER SAMPLES == 243

2.      NUMBER SAMPLES < 243
2.a.    number samples < minimum_number_frames
2.b.    minimum_number_frames <= number samples < 243 - 2*number_cutoff_frames
2.c.    243 - 2*number_cutoff_frames <= number samples < 243 

3.      NUMBER SAMPLES > 243
3.a.    243 < number samples <  243 + 2*number_cutoff_frames
3.b.    243 + 2*number_cutoff_frames <= number samples
'''

number_cutoff_frames = 30
minimum_number_frames = 50
sample_frames = 243
dataset_generator = DatasetClampDivideStrategy(number_cutoff_frames)

sample_1 = np.random.random((sample_frames, 17, 3))

sample_2a = np.random.random((np.random.randint(0, minimum_number_frames), 17, 3))
sample_2b = np.random.random((np.random.randint(minimum_number_frames, sample_frames - 2 * number_cutoff_frames), 17, 3))
sample_2c = np.random.random((np.random.randint(sample_frames - 2 * number_cutoff_frames, sample_frames), 17, 3))

sample_3a = np.random.random((np.random.randint(sample_frames, sample_frames + 2 * number_cutoff_frames), 17, 3))
sample_3b = np.random.random((np.random.randint(sample_frames + 2 * number_cutoff_frames, int(1e4)), 17, 3))


sample_1_motionbert = dataset_generator.get_samples(sample_1)

sample_2a_motionbert = dataset_generator.get_samples(sample_2a)
sample_2b_motionbert = dataset_generator.get_samples(sample_2b)

sample_3a_motionbert = dataset_generator.get_samples(sample_3a)
sample_3b_motionbert = dataset_generator.get_samples(sample_3b)
sample_3c_motionbert = dataset_generator.get_samples(sample_3c)
