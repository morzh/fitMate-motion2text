import json
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from visualization_utilities import *
import colors_styles


folder_root = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions'
folder_source_video = 'source'
folder_annotations = 'annotations'
folder_vis_skeleton = 'vis_whole_skeleton'
show_frame_results = False

files = [f for f in os.listdir(os.path.join(folder_root, folder_source_video)) if os.path.isfile(os.path.join(folder_root, folder_source_video, f))]

for file in files:
    path_filename_annotations = os.path.join(folder_root, folder_annotations, file+'.json')
    path_filename_source_video = os.path.join(folder_root, folder_source_video, file)
    path_filename_vis_skeleton_video = os.path.join(folder_root, folder_vis_skeleton, file)
    if not os.path.exists(path_filename_annotations):
        continue

    with open(path_filename_annotations, 'r') as f:
        video_data = json.load(f)
        video_shape = (video_data['height'], video_data['width'])
        annotations_frames_number = video_data['frame_count']
        annotations_pct = video_data['annotations_PCT']

    poses_coordinates = [annotation['pose'] for annotation in annotations_pct]
    poses_frames = [annotation['frame'] for annotation in annotations_pct]
    poses_dict = {annotation['frame']: np.array(annotation['pose'])[:, :2] for annotation in annotations_pct}

    poses_coordinates = np.array(poses_coordinates)
    poses_coordinates = poses_coordinates[:, :, :2]

    poses_minimum_x = np.min(poses_coordinates[:, :, 0])
    poses_maximum_x = np.max(poses_coordinates[:, :, 0])
    poses_minimum_y = np.min(poses_coordinates[:, :, 1])
    poses_maximum_y = np.max(poses_coordinates[:, :, 1])

    new_frame_top_left = np.array([min(poses_minimum_x, 0), min(poses_minimum_y, 0)])
    new_frame_top_left = np.floor(new_frame_top_left).astype(np.int32)
    new_frame_right_bottom = np.array([max(poses_maximum_x, video_shape[1]), max(poses_maximum_y, video_shape[0])])
    new_frame_right_bottom = np.ceil(new_frame_right_bottom).astype(np.int32)
    new_frame_resolution = new_frame_right_bottom - new_frame_top_left

    frame_extended = 255 * np.ones((new_frame_resolution[1], new_frame_resolution[0], 3)).astype(np.uint8)
    video_capture_source = cv2.VideoCapture(path_filename_source_video)
    video_frames_number = int(video_capture_source.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video_capture_source.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_capture_target = cv2.VideoWriter(path_filename_vis_skeleton_video, fourcc, video_fps, (new_frame_resolution[0], new_frame_resolution[1]))

    index_annotation_frame = 0
    for index_frame in range(video_frames_number):
        ret, frame = video_capture_source.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_extended_current = np.copy(frame_extended)
        frame_extended_current[-new_frame_top_left[0]: -new_frame_top_left[0] + video_shape[0], -new_frame_top_left[1]: -new_frame_top_left[1] + video_shape[1], :] = frame

        if index_frame in poses_frames:
            pose_current = poses_dict[index_frame] - new_frame_top_left
            pose_current = np.hstack((pose_current, np.ones((pose_current.shape[0], 1))))
            pose_current = np.expand_dims(pose_current, axis=0)
            frame_extended_current = draw_skeleton(frame_extended_current, pose_current, colors_styles.chunhua_style)

        if show_frame_results and file == 'bored man clapping hands over green screen.mp4':
            plt.imshow(frame_extended_current)
            plt.scatter(pose_current[0, :, 0], pose_current[0, :, 1], c='r')
            plt.show()
        video_capture_target.write(frame_extended_current)

    video_capture_source.release()
    video_capture_target.release()

