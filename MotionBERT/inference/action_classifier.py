import os
import sys
from typing import Tuple
import pickle5
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from lib.model.model_action import ActionNet

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class ActionClassifier:
    LABEL_FPATH = str(PROJECT_ROOT / "data" / "action" / "ntu_actions.txt")

    def __init__(self, args):
        self._args = args
        model_backbone = load_backbone(args)
        self.model = ActionNet(backbone=model_backbone,
                               dim_rep=args.dim_rep,
                               num_classes=args.action_classes,
                               dropout_ratio=args.dropout_ratio,
                               version=args.model_version,
                               hidden_dim=args.hidden_dim,
                               num_joints=args.num_joints)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        with open(self.LABEL_FPATH, 'r') as file:
            self.labels = [label.split(".")[1][1:] for label in file.readlines()]

    def inference(self, skeletons: np.ndarray) -> Tuple[np.ndarray, list]:
        self.check_data_shape(skeletons)
        skeletons = self.check_data_dimensions(skeletons)

        skeletons = torch.tensor(skeletons)
        with torch.no_grad():
            if torch.cuda.is_available():
                skeletons = skeletons.cuda()
            output = self.model(skeletons)  # (N, num_classes)
        return output, self._idx2lbl(output)

    def _idx2lbl(self, label_idx: torch.tensor) -> list:
        return self.labels[label_idx.argmax()]

    def check_data_shape(self, data: np.ndarray):
        if data.shape[-3] != self._args.clip_len:
            right_shape = (0, 2, self._args.clip_len, self._args.num_joints, 3)[-len(data.shape):]
            raise ValueError(f"Wrong shape of data. \ndata shape: {data.shape} != {right_shape} ")

    @staticmethod
    def check_data_dimensions(data):
        for i in range(5 - len(data.shape)):
            data = data[None]
        return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/action/MB_train_NTU60_xsub.yaml",
                        help="Path to the config file.")
    parser.add_argument("--skeletons_fpath", type=str, default="",
                        help="Path to data for classification.")
    parser.add_argument("--save_fpath", type=str, default="run/classification_results.txt",
                        help="Path to file for save classification results.")
    opts = parser.parse_args()
    return opts


def run_example(model):
    skeletons = np.load(str(PROJECT_ROOT / "data" / "action_example.npy"))
    skeletons = skeletons[:243]
    probs, labels = model.inference(skeletons)
    print(labels)


def run(fpath: str, result_fpath: str, model: ActionClassifier) -> None:
    data = read_pickle(fpath)
    concat_data = np.concatenate(list(data.values()))[None].swapaxes(0, 1)
    probs, labels = model.inference(concat_data)
    pass


def read_pickle(fpath):
    with open(str(PROJECT_ROOT / fpath), 'rb') as file:
        data = pickle5.load(file)
    return data


if __name__ == '__main__':
    opts = parse_args()
    args = get_config(PROJECT_ROOT / opts.config)
    model = ActionClassifier(args)
    if not opts.skeletons_fpath:
        run_example(model)
    else:
        run(opts.skeletons_fpath, opts.save_fpath, model)
