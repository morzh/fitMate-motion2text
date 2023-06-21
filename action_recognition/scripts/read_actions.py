from pathlib import Path
from tqdm import tqdm
from settings import webcam_options
from tools.skeleton_reader import SkeletonReader
from tools.video_processing import VideoReader, VideoWriter
from tools.interpolation import Interpolation

PROJECT_ROOT = Path(__file__).parent.parent


def show_mediapipe_processing(source):
    video_reader = VideoReader(source, webcam_options)
    skeleton_reader = SkeletonReader()

    for image, frame_idx in video_reader.image_generator():
        skeleton = skeleton_reader.get_skeleton(image)
        image = skeleton_reader.draw_pose_points()
        video_reader.show_frame(image)
        skeleton_reader.plot_3d_skeleton(skeleton)


def write_mediapipe_video(video_fpath: Path, save_dpath: Path):
    video_reader = VideoReader(video_fpath, webcam_options)
    skeleton_reader = SkeletonReader()
    save_dpath.mkdir(exist_ok=True, parents=True)
    save_fpath = save_dpath / (video_fpath.with_suffix(".mp4").name)
    with VideoWriter(save_fpath, fps=video_reader.fps) as video_writer:
        for image, frame_idx in video_reader.image_generator():
            print(frame_idx)
            skeletons = skeleton_reader.get_skeleton(image)
            image = skeleton_reader.draw_pose_points()
            # video_reader.show_frame(image)
            video_writer.write_frame(image)


def write_3d_animation(skeletons_fpath: Path, save_fpath: Path, interpolation=True, fps=30):
    skeleton_reader = SkeletonReader()
    skeletons = skeleton_reader.read_skeletons_from_file(str(skeletons_fpath))
    if interpolation:
        skeletons = Interpolation().interpolate_skeletons(skeletons)

    with VideoWriter(save_fpath, fps=fps) as video_writer:
        for skeleton in tqdm(skeletons, desc=f"Write video: {save_fpath}"):
            skeleton_reader.plot_3d_skeleton(skeleton, show=False)
            image = skeleton_reader.get_3d_image(640)
            video_writer.write_frame(image)

    print(f"Video saved: {save_fpath}")


if __name__ == '__main__':
    # test_video = PROJECT_ROOT / "dataset/from_team/streams/aleksey_train.mp4"
    test_video = PROJECT_ROOT / "dataset/from_team/leg_arm/IMG_4400 (1).mov"
    # test_video = PROJECT_ROOT / "dataset/from_team/pullDown/1036 || Sergey | pullDown.mov"
    webcam_idx = 0
    save_fpath = PROJECT_ROOT / "results" / "videos"
    show_mediapipe_processing(test_video)
    write_mediapipe_video(test_video, save_fpath)
