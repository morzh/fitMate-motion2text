# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import matplotlib.pyplot as plt
import mmcv
import torch
import numpy as np
import cv2

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

# from mmpose.apis import multi_gpu_test, single_gpu_test
# from mmpose.datasets import build_dataloader, build_dataset

from models import build_posenet

import configs.pct_large_classifier as cfg


try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    model = build_posenet(cfg.model)
    # model = fuse_conv_bn(model)

    img = cv2.imread('/home/morzh/work/fitMate/repositories/PCT/test_images/test_pose_0002.png')
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img /= 255.0
    plt.imshow(img)
    plt.show()

    img = 2.0 * img - 1.0
    images_tensor = torch.from_numpy(img)
    images_tensor = images_tensor.permute(2, 0, 1)
    images_tensor = torch.unsqueeze(images_tensor, dim=0)

    '''
    img_metas (list(dict)): Information about data augmentation
    By default this includes:
    - "image_file: path to the image file
    - "center": center of the bbox
    - "scale": scale of the bbox
    - "rotation": rotation of the bbox
    - "bbox_score": score of bbox
    '''

    results = []
    img_metas = dict(
        image_file='/home/morzh/work/fitMate/repositories/PCT/test_images/test_pose_0002.png',
        center=(127, 127),
        scale=1.0,
        rotation=0,
        bbox_score=0.98)

    with torch.no_grad():
        result = model(images_tensor, img_metas=[img_metas], return_loss=False)
        print(result['preds'][0][:, :2])
        plt.imshow(0.5*(img + 1))
        plt.scatter(result['preds'][0][:, 0], result['preds'][0][:, 1])
        plt.show()
    results.append(result)


if __name__ == '__main__':
    main()
