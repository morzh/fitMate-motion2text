# Copyright (c) OpenMMLab. All rights reserved.
# The visualization code is from HRNet(https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

import os, sys
import warnings
from argparse import ArgumentParser

sys.path.append('/home/morzh/work/fitMate/repositories/PCT')

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import mmcv
from mmcv.runner import load_checkpoint
from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results)
from mmpose.datasets import DatasetInfo
from models import build_posenet
from colors_styles import *

# try:
from mmdet.apis import inference_detector, init_detector
has_mmdet = True
# except (ImportError, ModuleNotFoundError):
#     has_mmdet = False


chunhua_style = ColorStyle(color2, link_pairs2, point_color2)


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
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(data_numpy[:,:,::-1])
    bk.set_zorder(-1)
    
    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(17,-1)
        joints_dict = map_joint_dict(dt_joints)
        
        # stick 
        for k, link_pair in enumerate(chunhua_style.link_pairs):
            if k in range(11,16):
                lw = thickness
            else:
                lw = thickness * 2

            line = mlines.Line2D(
                    np.array([joints_dict[link_pair[0]][0],
                                joints_dict[link_pair[1]][0]]),
                    np.array([joints_dict[link_pair[0]][1],
                                joints_dict[link_pair[1]][1]]),
                    ls='-', lw=lw, alpha=1, color=link_pair[2],)
            line.set_zorder(0)
            ax.add_line(line)

        # black ring
        for k in range(dt_joints.shape[0]):
            if k in range(5):
                radius = thickness
            else:
                radius = thickness * 2

            circle = mpatches.Circle(tuple(dt_joints[k,:2]), 
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
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
    plt.margins(0,0)

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


def main():
    det_config = 'cascade_rcnn_x101_64x4d_fpn_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
    pose_config = '../configs/pct_large_classifier.py'
    pose_checkpoint = '../weights/pct/swin_large.pth'
    det_cat_id = 1
    bbox_threshold = 0.6
    line_thickness = 2

    det_model = init_detector(det_config, det_checkpoint, device='cuda')
    pose_model = init_pose_model(pose_config, pose_checkpoint, device='cuda')

    dataset = 'TopDownCocoDataset'
    source_folder = '../test_images'
    target_folder = '../test_images_out'
    image_name = 'test_pose_0011.png'
    source_image_filepath = os.path.join(source_folder, image_name)
    target_image_filepath = os.path.join(target_folder, 'vis_' + image_name)

    # tests a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, source_image_filepath)

    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mmdet_results, det_cat_id)
    person_bbox = 0
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        source_image_filepath,
        person_results,
        bbox_thr=bbox_threshold,
        format='xyxy',
        dataset=dataset,
        dataset_info=None,
        return_heatmap=False,
        outputs=None)

    vis_pose_result(source_image_filepath, pose_results, thickness=line_thickness, out_file=target_image_filepath)


if __name__ == '__main__':
    main()
