import time
import sys
import cv2
import numpy as np
# import ffmpeg
# from moviepy.editor import VideoFileClip
# from tinytag import TinyTag
# import exifread
import torch
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patches as patches

import mmcv
from mmcv.runner import load_checkpoint
from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results, inference)
from mmpose.utils.hooks import OutputHook
from mmpose.datasets import DatasetInfo
from models import build_posenet
from colors_styles import *
from mmdet.apis import inference_detector, init_detector

has_mmdet = True

cfg = dict(
    filename_source='/media/anton/78c429b5-289a-4928-ba55-218ce513ebf5/home/morzh/Videos/IMG_2920.MOV',
    filename_target='/media/anton/78c429b5-289a-4928-ba55-218ce513ebf5/home/morzh/Videos/IMG_2920_bboxes.mp4',
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


def bbox_xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict


def vis_pose_result(image_name, pose_results, thickness, out_file):
    data_numpy = cv2.imread(image_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    h = data_numpy.shape[0]
    w = data_numpy.shape[1]

    # Plot
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.subplot(1, 1, 1)
    bk = plt.imshow(data_numpy[:, :, ::-1])
    bk.set_zorder(-1)

    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(17, -1)
        joints_dict = map_joint_dict(dt_joints)

        # stick
        for k, link_pair in enumerate(chunhua_style.link_pairs):
            if k in range(11, 16):
                lw = thickness
            else:
                lw = thickness * 2

            line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0],
                          joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1],
                          joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=link_pair[2], )
            line.set_zorder(0)
            ax.add_line(line)

        # black ring
        for k in range(dt_joints.shape[0]):
            if k in range(5):
                radius = thickness
            else:
                radius = thickness * 2

            circle = mpatches.Circle(tuple(dt_joints[k, :2]),
                                     radius=radius,
                                     ec='black',
                                     fc=chunhua_style.ring_color[k],
                                     alpha=1,
                                     linewidth=1)
            circle.set_zorder(1)
            ax.add_patch(circle)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(out_file, format='png', bbox_inches='tight', dpi=100)
    plt.close()


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
    # save the config in the model for convenience
    model.video_settigns = config
    model.to(device)
    model.eval()
    return model


def draw_bboxes(img, bboxes_xywh, threshold=0.5, color=(255, 90, 90), thickness=2):
    if len(bboxes_xywh) == 0:
        return img
    bboxes_selected = []
    for bbox in bboxes_xywh:
        if bbox[4] > threshold:
            bboxes_selected.append(bbox)
        '''
        else:
            bbox = bbox.astype(int)
            img = cv2.line(img, (bbox[0],  bbox[1]), (bbox[0] +  bbox[2],   bbox[1]), color=(0, 0, 0), thickness=thickness)
            img = cv2.line(img, (bbox[0] + bbox[2],   bbox[1]), (bbox[0] +  bbox[2], bbox[1] + bbox[3]), color=(0, 0, 0), thickness=thickness)
            img = cv2.line(img, (bbox[0] + bbox[2],   bbox[1] +  bbox[3]), (bbox[0], bbox[1] + bbox[3]), color=(0, 0, 0), thickness=thickness)
            img = cv2.line(img, (bbox[0],  bbox[1] +  bbox[3]), (bbox[0],   bbox[1]), color=(0, 0, 0), thickness=thickness)
        '''

    if len(bboxes_selected) == 0:
        return img

    bbox_largest = bboxes_selected[0]
    largest_area = bbox_largest[2] * bbox_largest[3]
    for idx in range(1, len(bboxes_selected)):
        area_current = bboxes_selected[idx][2] * bboxes_selected[idx][3]
        if area_current > largest_area:
            largest_area = area_current
            bbox_largest = bboxes_selected[idx]

    for bbox in bboxes_selected:
        bbox = bbox.astype(int)
        img = cv2.line(img, (bbox[0],  bbox[1]), (bbox[0] +  bbox[2],   bbox[1]), color=color, thickness=thickness)
        img = cv2.line(img, (bbox[0] + bbox[2],   bbox[1]), (bbox[0] +  bbox[2], bbox[1] + bbox[3]), color=color, thickness=thickness)
        img = cv2.line(img, (bbox[0] + bbox[2],   bbox[1] +  bbox[3]), (bbox[0], bbox[1] + bbox[3]), color=color, thickness=thickness)
        img = cv2.line(img, (bbox[0],  bbox[1] +  bbox[3]), (bbox[0],   bbox[1]), color=color, thickness=thickness)

    bbox_largest = bbox_largest.astype(int)
    img = cv2.line(img, (bbox_largest[0],  bbox_largest[1]), (bbox_largest[0] +  bbox_largest[2],   bbox_largest[1]), color=(0, 255, 0), thickness=thickness)
    img = cv2.line(img, (bbox_largest[0] + bbox_largest[2],   bbox_largest[1]), (bbox_largest[0] +  bbox_largest[2], bbox_largest[1] + bbox_largest[3]), color=(0, 255, 0), thickness=thickness)
    img = cv2.line(img, (bbox_largest[0] + bbox_largest[2],   bbox_largest[1] +  bbox_largest[3]), (bbox_largest[0], bbox_largest[1] + bbox_largest[3]), color=(0, 255, 0), thickness=thickness)
    img = cv2.line(img, (bbox_largest[0],  bbox_largest[1] +  bbox_largest[3]), (bbox_largest[0],   bbox_largest[1]), color=(0, 255, 0), thickness=thickness)

    return img



chunhua_style = ColorStyle(color2, link_pairs2, point_color2)

show_frame_results = False


def bounding_boxes_process(bounding_boxes, threshold):
    number_boxes = len(bounding_boxes)

    if number_boxes == 0:
        return None

    boxes_selected = []

    for index in range(number_boxes):
        if bounding_boxes[index]['bbox'][4] > threshold:
            boxes_selected.append(bounding_boxes[index]['bbox'])

    if len(boxes_selected) == 0:
        return None

    largest_bbox = boxes_selected[0]
    largest_bbox_area = largest_bbox[2] * largest_bbox[3]

    for index in range(1, len(boxes_selected)):
        current_bbox_area = boxes_selected[index][2] * boxes_selected[index][3]
        if current_bbox_area > largest_bbox_area:
            largest_bbox = boxes_selected[index]
            largest_bbox_area = current_bbox_area

    return largest_bbox


def bboxes_dict2ndarray(persons_bboxes):
    bboxes = np.empty((0, 5))
    for index in range(len(persons_bboxes)):
        bboxes = np.vstack((bboxes, persons_bboxes[index]['bbox']))
    return bboxes


def main():
    det_model = init_detector(cfg['det_config'], cfg['det_checkpoint'], device='cuda')
    capture_source = cv2.VideoCapture(cfg['filename_source'])
    if not capture_source.isOpened():
        print("Cannot open camera")
        exit()

    fps = capture_source.get(cv2.CAP_PROP_FPS)
    frame_count = capture_source.get(cv2.CAP_PROP_FRAME_COUNT)
    cap_width = int(capture_source.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    cap_height = int(capture_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    show_scaled_images = False

    index_frame = 0
    index_frame_buffer = 0

    width_scaled = int(cap_width * cfg['scale_percent'] / 100.0)
    height_scaled = int(cap_height * cfg['scale_percent'] / 100.0)
    frames_resized_buffer = np.empty((0, cap_height, cap_width, 3), dtype=np.uint8)
    # frames_resized_buffer = []

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    capture_target = cv2.VideoWriter(cfg['filename_target'], fourcc, cfg['fps_target'], (cap_width, cap_height))

    print('number of frames:', frame_count)
    time_start = time.time()
    while True:
        percentage = index_frame / (frame_count - 1)
        time.sleep(1e-5)
        sys.stdout.write("\r%f%%" % percentage)
        sys.stdout.flush()

        ret, frame = capture_source.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mmdet_results = inference_detector(det_model, frame)
        persons_bboxes = process_mmdet_results(mmdet_results, cfg['det_cat_id'])
        persons_bboxes = bboxes_dict2ndarray(persons_bboxes)
        persons_bboxes = bbox_xyxy2xywh(persons_bboxes)

        frame_visualization = draw_bboxes(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), persons_bboxes)

        if show_frame_results:
            cv2.imshow('Bbox', frame_visualization)
            cv2.waitKey(-1)

        capture_target.write(frame_visualization)

        index_frame += 1

    time_end = time.time()

    capture_source.release()
    capture_target.release()

    cv2.destroyAllWindows()

    print('processing time:', time_end - time_start, 'seconds')


if __name__ == '__main__':
    main()
