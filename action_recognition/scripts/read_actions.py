from action_recognition.tools.skeleton_reader import SkeletonReader
from action_recognition.tools.video_processing import VideoReader, VideoWriter
from pathlib import Path
from action_recognition.settings import webcam_options

PROJECT_ROOT = Path(__file__).parent.parent


def show_mediapipe_processing(source):
    video_reader = VideoReader(source, webcam_options)
    skeleton_reader = SkeletonReader()

    for image, frame_idx in video_reader.image_generator():
        frame_skeletons = skeleton_reader.get_skeleton(image)
        image = skeleton_reader.draw_pose_points()
        video_reader.show_frame(image)


def write_mediapipe_video(video_fpath: Path, save_dpath: Path):
    video_reader = VideoReader(video_fpath, webcam_options)
    skeleton_reader = SkeletonReader()
    save_dpath.mkdir(exist_ok=True, parents=True)
    save_fpath = save_dpath / (video_fpath.with_suffix(".mp4").name)
    with VideoWriter(save_fpath, fps=video_reader.fps) as video_writer:
        for image, frame_idx in video_reader.image_generator():
            print(frame_idx)
            frame_skeletons = skeleton_reader.get_skeleton(image)
            image = skeleton_reader.draw_pose_points()
            # video_reader.show_frame(image)
            video_writer.write_frame(image)


if __name__ == '__main__':
    # test_video = PROJECT_ROOT / "dataset/from_team/streams/aleksey_train.mp4"
    test_video = PROJECT_ROOT / "dataset/from_team/leg_arm/IMG_4400 (1).mov"
    # test_video = PROJECT_ROOT / "dataset/from_team/pullDown/1036 || Sergey | pullDown.mov"
    webcam_idx = 0
    save_fpath = PROJECT_ROOT / "results" / "videos"
    show_mediapipe_processing(test_video)
    write_mediapipe_video(test_video, save_fpath)