import json
import os
import numpy as np
import pickle
import torch

folder_dataset = '/home/anton/work/fitMate/datasets/ActionDatasets/TestActions'
claps_dataset_filename = 'classification_results_ours.pkl'

with open(os.path.join(folder_dataset, claps_dataset_filename), 'rb') as f:
    dataset_results = pickle.load(f)

print(dataset_results)