from typing import Union, Dict
from pathlib import Path
from threading import Thread
import numpy as np
import cv2
from screeninfo import get_monitors


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
        if not status:
            raise Exception(f"Source path '{self.source}' error, video can't be read.")
        self.frame_img = frame_img
        self.frame_idx = 0
        self._close = False
        self._frame_name = self._build_frame_name()
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

    def show_frame(self,
                   image: np.ndarray = None,
                   frame_name: str = None,
                   show_full_frame: bool = True):
        if image is None:
            image = self.frame_img
        if frame_name is None:
            frame_name = self._frame_name

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
                h = int(sys_monitors[0].height * k * 0.8)
                w = int(sys_monitors[0].width * k * 0.8)

        return w, h

    def _build_frame_name(self):
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
