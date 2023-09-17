import os
import time
import sys
import json
import matplotlib.pyplot as plt
import colors_styles

from numpy_json_encoder import NumpyJsonEncoder
from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results, inference)
from mmpose.utils.hooks import OutputHook
from init_pose_model import init_pose_model
from mmdet.apis import inference_detector, init_detector
from visualization_utilities import *

plt.switch_backend('Agg')
has_mmdet = True


dataset_settings = dict(
    folder_root='/home/anton/work/fitMate/datasets/ActionDatasets/TestActions/',
    folder_source='source',
    folder_annotations='annotations',
    folder_visualization='visualizations',
    video_files_extensions=['mp4', 'MP4', 'mov', 'MOV']
)

pose_estimation_settings = dict(
    fps_target=30.0,
    scale_percent=50,
    images_batch_size=2,
    det_config='./cascade_rcnn_x101_64x4d_fpn_coco.py',
    det_checkpoint='https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth',
    pose_config='../configs/pct_large_classifier.py',
    pose_checkpoint='../weights/pct/swin_large.pth',
    det_cat_id=1,
    bbox_threshold=0.5,
)

visualization_settings = dict(
    show_frame_results=False,
    save_visualization=True
)


def process_video_file(video_filename, det_model, pose_model, dataset_settings, pose_settings, viz_settings):
    print('-'*100)
    print(f"{video_filename}")
    capture_source = cv2.VideoCapture(os.path.join(dataset_settings['folder_root'], dataset_settings['folder_source'], video_filename))

    if not capture_source.isOpened():
        print(f"Cannot open video file {video_filename}")
        exit()

    fps = capture_source.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture_source.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_width = int(capture_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(capture_source.get(cv2.CAP_PROP_FRAME_HEIGHT))

    index_frame = 0
    if viz_settings['save_visualization']:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_filename_visualization = os.path.join(dataset_settings['folder_root'], dataset_settings['folder_visualization'], video_filename)
        capture_target = cv2.VideoWriter(video_filename_visualization, fourcc, pose_settings['fps_target'], (cap_width, cap_height))

    video_data = {'filename': video_filename, 'width': cap_width, 'height': cap_height, 'fps': fps, 'frame_count': frame_count}

    annotations = []

    print('number of frames:', frame_count)
    time_start = time.time()
    while index_frame < frame_count:
        percentage = 100 * index_frame / (frame_count - 1)
        time.sleep(1e-5)
        sys.stdout.write("\r%f%%" % percentage)
        sys.stdout.flush()

        ret, frame = capture_source.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_resized_buffer = np.expand_dims(frame, axis=0)
        mmdet_results = inference_detector(det_model, frame)
        persons_bboxes = process_mmdet_results(mmdet_results, pose_settings['det_cat_id'])
        persons_bboxes = bboxes_dict2ndarray(persons_bboxes)
        persons_bboxes = bbox_xyxy2xywh(persons_bboxes)
        bounding_box = get_largest_bbox(persons_bboxes, pose_settings['bbox_threshold'])

        if bounding_box is None:
            capture_target.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index_frame += 1
            continue

        person_bbox = np.expand_dims(bounding_box, axis=0)
        # print('person_bbox', person_bbox)

        with OutputHook(pose_model, outputs=None, as_tensor=True) as h:
            # poses is results['pred'] # N x 17x 3
            poses, heatmap = inference._inference_single_pose_model(pose_model, frames_resized_buffer, person_bbox, return_heatmap=False, use_multi_frames=False)

        frame_visualization = draw_bbox(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), person_bbox[0])
        frame_visualization = draw_skeleton(frame_visualization, poses, colors_styles.chunhua_style)
        frame_data = {'frame': index_frame, 'bbox': person_bbox[0].astype(np.longdouble).round(2), 'pose': poses[0].astype(np.longdouble).round(2)}

        if viz_settings['show_frame_results']:
            cv2.imshow('Bbox', frame_visualization)
            cv2.waitKey(-1)

        if viz_settings['save_visualization']:
            capture_target.write(frame_visualization)

        annotations.append(frame_data)
        index_frame += 1

    video_data['annotations_PCT'] = annotations

    time_end = time.time()
    cv2.destroyAllWindows()
    capture_source.release()
    if viz_settings['save_visualization']:
        capture_target.release()

    with open(os.path.join(dataset_settings['folder_root'], dataset_settings['folder_annotations'], video_filename + '.json'), 'w') as outfile:
        json.dump(video_data, outfile, indent=4, ensure_ascii=False, cls=NumpyJsonEncoder)

    print(';  processing time:', time_end - time_start, 'seconds')
    print('\n')


def main():
    det_model = init_detector(pose_estimation_settings['det_config'], pose_estimation_settings['det_checkpoint'], device='cuda')
    pose_model = init_pose_model(pose_estimation_settings['pose_config'], pose_estimation_settings['pose_checkpoint'], device='cuda')

    folder_videos = os.path.join(dataset_settings['folder_root'], dataset_settings['folder_source'])
    files = [f for f in os.listdir(folder_videos) if os.path.isfile(os.path.join(folder_videos, f)) and f.endswith(tuple(dataset_settings['video_files_extensions']))]
    for file in files:
        process_video_file(file, det_model, pose_model, dataset_settings, pose_estimation_settings, visualization_settings)


if __name__ == '__main__':
    main()
