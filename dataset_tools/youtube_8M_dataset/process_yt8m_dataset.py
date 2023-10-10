import json
import os
import numpy as np
import requests
import tensorflow as tf
from tensorflow.data import Dataset, TFRecordDataset


YT8M_ID = 161  # gym label
DATASET_TYPE: str = 'train'
# DATASET_TYPE: str = 'validate'
ERROR_TAG = '<Error>'

# path_dataset_root = '/home/anton/work/fitMate/datasets/yt8m/segment_rated_frame_level'
path_dataset_root = '/home/anton/Downloads/YouTube-8M/data/video'
path_subfolder =  DATASET_TYPE
filename_json = ''.join([DATASET_TYPE, '_youtube_ids', '.json'])
filename_extension = 'tfrecord'
url_base = 'data.yt8m.org/2/j/i/'

folder_records = os.path.join(path_dataset_root, path_subfolder)
files = [f for f in os.listdir(folder_records) if os.path.isfile(os.path.join(folder_records, f)) and f.endswith(filename_extension)]
youtube_videos_ids = []

for file in files:
    file_path_name = os.path.join(folder_records, file)
    raw_dataset = tf.data.TFRecordDataset(file_path_name)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        video_labels = example.features.feature['labels'].int64_list.value
        segment_labels = example.features.feature['segment_labels'].int64_list.value
        if YT8M_ID in video_labels or YT8M_ID in segment_labels:
            dataset_video_id = example.features.feature['id'].bytes_list.value[0]
            dataset_video_id = dataset_video_id.decode('ascii')
            video_url_js = ''.join(['https://', url_base, dataset_video_id[:2], '/', dataset_video_id, '.js'])
            try:
                response = requests.get(video_url_js, verify=False)
                page_source = response.content.decode('ascii')
                if ERROR_TAG not in page_source:
                    youtube_video_id = page_source[10:-3]
                    youtube_videos_ids.append(youtube_video_id)
            except:
                continue

# gym_youtube_videos_json_string = json.dumps(youtube_videos_ids)
with open(os.path.join(path_dataset_root, filename_json), 'w') as f:
    json.dump(youtube_videos_ids, f, indent=4)
    
print('Number youtube videos with label', YT8M_ID, 'is', len(youtube_videos_ids))