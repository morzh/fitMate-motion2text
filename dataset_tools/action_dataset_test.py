import os
import json
import numpy as np
import pickle
import cv2
import vis_tools.colors_styles as colors_styles

from vis_tools.visualization_utilities import *


folder_dataset = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions'
folder_videos_source = 'source'
claps_dataset_filename = 'claps_youtube.pkl'

with open(os.path.join(folder_dataset, claps_dataset_filename), 'rb') as f:
    dataset = pickle.load(f)

for video_filename, annotation in dataset.items():
    file_pathname_video = os.path.join(folder_dataset, folder_videos_source, video_filename)
    video_capture_source = cv2.VideoCapture(file_pathname_video)

    fps = video_capture_source.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture_source.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_width = int(video_capture_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(video_capture_source.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = video_capture_source.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    draw_skeleton(frame, annotation['pose_pct'], colors_styles.chunhua_style, vis_scores=True)



