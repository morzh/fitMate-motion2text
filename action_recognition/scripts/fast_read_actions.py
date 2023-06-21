from pathlib import Path

from settings import webcam_options
from tools.fast_skeleton_reader import SkeletonReader
from tools.video_processing import VideoReader

PROJECT_ROOT = Path(__file__).parent


def run(source, n_process=4):
    if source.isdigit():
        source = int(source)
    video_reader = VideoReader(source, webcam_options)
    skeleton_reader = SkeletonReader(n_process)
    skeletons = []
    for image, frame_id in video_reader.image_generator():
        skeleton_reader.put_image(image, frame_id)
        results = skeleton_reader.ger_results()
        skeletons.extend(results)
        video_reader.show_frame(image)


if __name__ == '__main__':
    test_video = PROJECT_ROOT / "dataset/from_team/pullDown/1036 || Sergey | pullDown.mov"
    # webcam_idx = 0
    # run(test_video)
    run(test_video)
