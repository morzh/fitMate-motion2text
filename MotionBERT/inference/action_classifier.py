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

RESULT_DIR = PROJECT_ROOT/"run"
RESULT_DIR.mkdir(exist_ok=True)


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
            self.labels = np.array([label.split(".")[1][1:] for label in file.readlines()])

    def inference(self, skeletons: np.ndarray, return_top_n: int = 5) -> Tuple[np.ndarray, list]:
        skeletons = self.update_data_dimensions(skeletons)
        self.check_data_shape(skeletons)
        skeletons = self.update_fourth_dim(skeletons)

        skeletons = torch.tensor(np.array(skeletons, dtype="float32"))
        with torch.no_grad():
            if torch.cuda.is_available():
                skeletons = skeletons.cuda()
            output = self.model(skeletons)  # (N, num_classes)
        return output, self._idx2lbl(output, return_top_n)

    def _idx2lbl(self, label_idx: torch.tensor, return_top_n: int) -> list:
        topk_values, topk_indices = label_idx.topk(return_top_n, sorted=True)
        topk_indices = topk_indices.to("cpu").numpy()
        return self.labels[topk_indices]

    def check_data_shape(self, data: np.ndarray):
        if data.shape[-3] != self._args.clip_len:
            raise ValueError(f"Wrong length in third dim. \ndata length: {data.shape[-3]} != {self._args.clip_len} ")

    @staticmethod
    def update_data_dimensions(data):
        for i in range(5 - len(data.shape)):
            data = data[None]
        return data

    @staticmethod
    def update_fourth_dim(data):
        if data.shape[-4] != 2:
            data = np.concatenate([data, np.zeros(data.shape)], axis=1)
        return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/action/MB_train_NTU60_xsub.yaml",
                        help="Path to the config file.")
    parser.add_argument("--skeletons_fpath", type=str, default="",
                        help="Path to data for classification.")
    parser.add_argument("--save_fpath", type=str, default=RESULT_DIR/"classification_results.pkl",
                        help="Path to file for save classification results.")
    opts = parser.parse_args()
    return opts


def run_example(model):
    skeletons = np.load(str(PROJECT_ROOT / "data" / "action_example.npy"))
    skeletons = skeletons[:243]
    probs, labels = model.inference(skeletons)
    print(labels)


def run(fpath: str, result_fpath: str, model: ActionClassifier, return_top_n=5) -> None:
    data = read_pickle(fpath)
    concat_data = np.concatenate(list(data.values()))[None].swapaxes(0, 1)
    probs, pred_labels = model.inference(concat_data, return_top_n)
    result = {}
    pos = 0
    for label, action in data.items():
        parts_in_action = action.shape[0]
        result[label] = (pred_labels[pos:pos + parts_in_action], probs[pos:pos + parts_in_action])
        pos += parts_in_action
    write_pickle(result_fpath, result)


def read_pickle(fpath):
    with open(str(PROJECT_ROOT / fpath), 'rb') as file:
        data = pickle5.load(file)
    return data


def write_pickle(fpath, data):
    with open(fpath, 'wb') as file:
        pickle5.dump(data, file)


if __name__ == '__main__':
    opts = parse_args()
    args = get_config(PROJECT_ROOT / opts.config)
    model = ActionClassifier(args)
    if not opts.skeletons_fpath:
        run_example(model)
    else:
        run(opts.skeletons_fpath, opts.save_fpath, model)
