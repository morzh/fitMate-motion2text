import json
import os

import numpy as np
import pickle


from dataset_from_video_annotations import DatasetFromVideoAnnotations


def main():
    folder_annotations = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions/annotations'
    folder_dataset = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions/motionBERT_dataset_PCT'
    filenames = [f for f in os.listdir(folder_annotations) if os.path.isfile(os.path.join(folder_annotations, f)) and f.endswith('json')]

    dataset_generator = DatasetFromVideoAnnotations()

    dataset = {}
    for filename in filenames:
        json_filepath = os.path.join(folder_annotations, filename)
        with open(json_filepath, 'r') as f:
            annotation = json.load(f)

        video_shape = (annotation['width'], annotation['height'])
        poses = [p['pose'] for p in annotation['annotations_PCT']]
        poses = np.array(poses)
        poses = dataset_generator.get_samples(poses)
        poses = dataset_generator.reassign_pct_joints_scores(poses, video_shape)
        poses = dataset_generator.normalize(poses, video_shape)

        dataset[filename] = poses

        pickle_filename = os.path.splitext(filename)[0] + '.pkl'
        with open(os.path.join(folder_dataset, pickle_filename), 'wb') as f:
            pickle.dump(poses, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
