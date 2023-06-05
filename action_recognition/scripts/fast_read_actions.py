from action_recognition.tools.fast_skeleton_reader import SkeletonReader
from action_recognition.tools.video_processing import VideoReader
from pathlib import Path
from action_recognition.settings import webcam_options
import click

PROJECT_ROOT = Path(__file__).parent


@click.command()
@click.option("-s", "--source", required=True,
              help="The path to a video file or webcam index")
def run(source):
    if source.isdigit():
        source = int(source)
    video_reader = VideoReader(source, webcam_options)
    skeleton_reader = SkeletonReader(6)
    r = []
    for image, frame_id in video_reader.image_generator():
        skeleton_reader.put_image(image, frame_id)
        results = skeleton_reader.ger_results()
        r.extend(results)
        video_reader.show_frame(image)


if __name__ == '__main__':
    # test_video = PROJECT_ROOT / "dataset/from_team/pullDown/1036 || Sergey | pullDown.mov"
    # webcam_idx = 0
    # run(test_video)
    run()
