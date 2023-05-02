from tools.skeleton_reader import SkeletonReader
from tools.video_reader import VideoReader
from pathlib import Path
from settings import webcam_options
import click

PROJECT_ROOT = Path(__file__).parent


@click.command()
@click.option("-s", "--source", required=True,
              help="The path to a video file or webcam index")
def run(source):
    if source.isdigit():
        source = int(source)
    video_reader = VideoReader(source, webcam_options)
    skeleton_reader = SkeletonReader()
    for image, frame_idx in video_reader.image_generator():
        frame_skeletons = skeleton_reader.get_skeleton(image)
        image = skeleton_reader.draw_pose_points()
        video_reader.show_frame(image)


if __name__ == '__main__':
    test_video = PROJECT_ROOT / "dataset/from_team/pullDown/1036 || Sergey | pullDown.mov"
    webcam_idx = 0
    run(test_video)