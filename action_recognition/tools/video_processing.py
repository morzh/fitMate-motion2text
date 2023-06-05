from typing import Union, Dict
from pathlib import Path
from threading import Thread
from tqdm import tqdm
import imageio
import numpy as np
import cv2
from screeninfo import get_monitors
from datetime import datetime


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
        self._thread_reader = None
        status, frame_img = self.capture.read()
        self.frame_time = datetime.now()
        self.frame_idx = 0
        if not status:
            raise Exception(f"Source path '{self.source}' error, video can't be read.")
        self.frame_img = frame_img
        self._close = False
        self._window_name = self._build_window_name()
        self.show_size = self._get_show_size(frame_img)

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
            self._run_realtime_img_reader()
            while self.read_frame():
                yield self.frame_img, self.frame_time
        else:
            while self.read_frame():
                yield self.frame_img, self.frame_idx

    def _run_realtime_img_reader(self):
        if type(self.source) is int:
            self._thread_reader = Thread(target=self.run_reader_thread, args=())
            self._thread_reader.start()

    def run_reader_thread(self):
        while self.is_alife():
            self.read_frame()

    def read_frame(self):
        status, img = self.capture.read()
        if img is None:
            self._close = True
        else:
            self.frame_idx += 1
            self.frame_img = img
            self.frame_time = datetime.now()
        return not self._close

    def is_alife(self):
        return not self._close

    def show_frame(self,
                   image: np.ndarray = None,
                   frame_name: str = None,
                   show_full_frame: bool = True):
        if image is None:
            image = self.frame_img
        if frame_name is None:
            frame_name = self._window_name

        if show_full_frame and self.show_size is not None:
            image = cv2.resize(image, self.show_size)

        cv2.imshow(frame_name, image)
        k = cv2.waitKey(1)
        if k == 113:  # 'q' key to stop
            self._close = True

    @staticmethod
    def _get_show_size(image):
        sys_monitors = get_monitors()
        h, w, _ = image.shape
        if sys_monitors is not None and sys_monitors:
            k = 1
            monitor_h = sys_monitors[0].height
            monitor_w = sys_monitors[0].width
            if h > monitor_h:
                k = monitor_h / h

            if w > monitor_w:
                kw = monitor_w / w
                if kw < k:
                    k = kw

            if k != 1:
                h = int(h * k * 0.9)
                w = int(w * k * 0.9)

        return w, h

    def _build_window_name(self):
        if type(self.source) is int:
            frame_name = f"Webcam: {self.source}. Press 'q' to close."
        else:
            frame_name = f"Video: {self.source.name}. Press 'q' to close."
        return frame_name

    def __del__(self):
        self._close = True
        self.capture.release()
        if self._thread_reader is not None:
            self._thread_reader.join()


class VideoWriter:
    def __init__(self, fpath: Path, fps: float = 30.0, verbose: bool = True):
        self._stream = imageio.get_writer(fpath, fps=fps)
        self._verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._stream.close()

    def write_frame(self, frame: np.ndarray):
        """Write single frame."""

        self._stream.append_data(frame)

    def write(self, frames: list[np.ndarray]):
        """Write list of frames."""

        stream = frames
        if self._verbose:
            stream = tqdm(frames, desc="Frames processing")

        for frame in stream:
            self.write_frame(frame)
