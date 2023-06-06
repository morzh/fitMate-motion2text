import time

import numpy as np

from tqdm import tqdm
from action_recognition.tools.fast_skeleton_reader import SkeletonReader
from action_recognition.tools.video_processing import VideoReader

from pathlib import Path
from action_recognition.settings import webcam_options


PROJECT_ROOT = Path(__file__).parent
RESULT_ROOT = Path(__file__).parent / "results"


def run():
    skeleton_reader = SkeletonReader(5)
    for video in (PROJECT_ROOT / "dataset" / "from_team").glob("**/*.*"):
        save_path = RESULT_ROOT / Path("/".join(video.parts[8:])+".npy")
        if save_path.is_file():
            print(f"Was skipped {str(save_path)}")
            continue
        try:
            video_reader = VideoReader(video, webcam_options)
        except Exception:
            continue

        save_path.parent.mkdir(exist_ok=True, parents=True)
        skeletons = []
        for image, frame_id in tqdm(video_reader.image_generator()):
            skeleton_reader.put_image(image, frame_id)
            results = skeleton_reader.ger_results()
            skeletons.extend(results)
        time.sleep(1)
        results = skeleton_reader.ger_results(True)
        skeletons.extend(results)
        skeletons = [frame_data[1] for frame_data in skeletons]
        np.save(str(save_path), np.array(skeletons, dtype=object))
        print(f"Was processed {str(save_path)}")
        skeleton_reader.result_history = []


if __name__ == '__main__':
    run()
