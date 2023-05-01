from typing import Union, Dict
from pathlib import Path
from threading import Thread
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).parent.parent


class VideoReader:
    def __init__(self, source: Union[int, Path] = 0, options: Dict = None):
        """
        source: webcam index or Path like object to video
        options: VideoCapture options for webcams. Available options: fps, width, height
        Current available settings of your webcam you can see with command: uvcdynctrl -f
        """
        self.source = source
        self.capture = cv2.VideoCapture(source if type(source) is int else str(source))
        self._set_capture_options(options)
        status, frame_img = self.capture.read()
        if not status:
            raise Exception(f"Source path '{self.source}' error, video can't be read.")
        self.frame_img = frame_img
        self.frame_idx = 0
        self._close = False
        self._thread_reader = None
        self._frame_name = self._build_frame_name()

    def _set_capture_options(self, options):
        if "fps" in options:
            self.capture.set(cv2.CAP_PROP_FPS, options["fps"])
        if "width" in options:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, options["width"])
        if "height" in options:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, options["height"])

    @property
    def fps(self):
        return self.capture.get(cv2.CAP_PROP_FPS)

    def image_generator(self):
        if type(self.source) is int:
            self._run_img_reader()
            while self.is_alife():
                yield self.frame_img, self.frame_idx
        else:
            while self.is_alife():
                self.read_frame()
                yield self.frame_img, self.frame_idx

    def _run_img_reader(self):
        if type(self.source) is int:
            self._thread_reader = Thread(target=self.run_reader_thread, args=())
            self._thread_reader.start()

    def run_reader_thread(self):
        while self.is_alife():
            self.read_frame()

    def read_frame(self):
        _, self.frame_img = self.capture.read()
        self.frame_idx += 1

    def is_alife(self):
        return self.frame_img is not None and not self._close

    def show_frame(self, image: np.ndarray = None, frame_name: str = None):
        if image is None:
            image = self.frame_img
        if frame_name is None:
            frame_name = self._frame_name

        cv2.imshow(frame_name, image)
        k = cv2.waitKey(1)
        if k == 113:  # 'q' key to stop
            self._close = True
        elif k == -1:
            pass
        else:
            print(k)

    def _build_frame_name(self):
        if type(self.source) is int:
            frame_name = f"Webcam: {self.source}. Press 'q' to close."
        else:
            frame_name = f"Video: {self.source.name}. Press 'q' to close."
        return frame_name

    def __del__(self):
        self._close = True
        if self._thread_reader is not None:
            self._thread_reader.join()


def test_run(source):
    options = {
        "fps": 30,
        "width": 1920,
        "height": 1080
    }

    reader = VideoReader(source, options)
    for frame, frame_idx in reader.image_generator():
        reader.show_frame()
        print(frame_idx)


if __name__ == "__main__":
    test_video = PROJECT_ROOT / "dataset/from_team/pullDown/1036 || Sergey | pullDown.mov"
    webcam_idx = 0
    test_run(source=webcam_idx)
