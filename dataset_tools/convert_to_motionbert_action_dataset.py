import json
import os
import numpy as np
import pickle
from dataset_from_video_annotations import DatasetFromVideoAnnotations


def main():
    folder_dataset = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions'
    folder_source_videos = 'source'
    folder_annotations = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions/annotations'
    claps_dataset_filename = 'claps_youtube.pkl'
    filenames = [f for f in os.listdir(folder_annotations) if os.path.isfile(os.path.join(folder_annotations, f)) and f.endswith('json')]

    dataset_generator = DatasetFromVideoAnnotations()

    dataset = {}
    for filename in filenames:
        print(filename)
        json_filepath = os.path.join(folder_annotations, filename)
        with open(json_filepath, 'r') as f:
            annotation = json.load(f)

        video_source_filename = annotation['filename']
        video_shape = (annotation['width'], annotation['height'])

        poses = [p['pose'] for p in annotation['annotations_PCT']]
        poses = np.array(poses)

        poses_frames = [p['frame'] for p in annotation['annotations_PCT']]
        poses_frames = np.array(poses_frames)

        poses = dataset_generator.reassign_pct_joints_scores(poses, video_shape)
        video_file_pathname = os.path.join(folder_dataset, folder_source_videos, video_source_filename)
        # dataset_generator.visualize_joints(poses, poses_frames, video_file_pathname, vis_joints_scores=True)
        poses = dataset_generator.get_samples(poses)
        poses = dataset_generator.normalize(poses, video_shape)

        dataset[os.path.splitext(filename)[0]] = poses.astype(np.float32)

    with open(os.path.join(folder_dataset, claps_dataset_filename), 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
