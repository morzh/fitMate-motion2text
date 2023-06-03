import time

import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Process, Queue
import queue

from threading import Thread

from settings import mediapipe_options

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class SkeletonReader:
    def __init__(self, n_process=3):
        self.result_history = []
        self.image_queue = Queue(maxsize=1)
        self.skeleton_queue = Queue()
        self.mp_processes = []
        self._thread_reader = None
        for i in range(n_process):
            p = Process(target=self.mp_process, args=(self.image_queue, self.skeleton_queue, i))
            p.start()
            self.mp_processes.append(p)
        self._thread_reader = Thread(target=self._read_results, args=())
        self._thread_reader.start()

    def mp_process(self, image_queue: Queue, skeleton_queue: Queue, process_idx: int):
        print(f"Process {process_idx} run")
        pose_model = mp.solutions.pose.Pose(**mediapipe_options)
        try:
            while True:
                image, image_time = image_queue.get(timeout=5)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_skeletons = pose_model.process(image)
                skeletons = self.mp_skeleton_to_np(mp_skeletons)
                skeleton_queue.put((skeletons, image_time))
        except queue.Empty:
            print(f"Process {process_idx} stop")
            exit(0)

    @staticmethod
    def mp_skeleton_to_np(mp_skeletons):
        if mp_skeletons.pose_landmarks is not None:
            np_results = np.array([(joint.x, joint.y, joint.z) for joint in
                                   list(mp_skeletons.pose_landmarks.landmark)],
                                  dtype="float32")
        else:
            np_results = None
        return np_results

    def put_image(self, image, image_time):
        self.image_queue.put((image, image_time))

    def _read_results(self):
        try:
            while True:
                skeletons, image_time = self.skeleton_queue.get(timeout=4)
                self.result_history.append((image_time, skeletons))
        except queue.Empty:
            exit(0)

    def ger_results(self, get_all=False):
        results = []
        if len(self.result_history) > 20:
            results = self.result_history
            self.result_history = []
            results = sorted(results)
            self.result_history = results[-10:] + self.result_history
        if not get_all:
            results = results[:-10]
        return results

    # def get_all_results(self):
    #     while True:
    #         status = []
    #         for process in self.mp_processes:
    #             status.append(process.is_alive())
    #         time.sleep(1)
    #         if all(status):
    #             break
    #     # results = [r[1] for r in sorted(self.result_history)]
    #     return sorted(self.result_history)

    @staticmethod
    def draw_pose_points(image, points):
        mp_drawing.draw_landmarks(
            image,
            points.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        return image

    def __del__(self):
        try:
            for process in self.mp_processes:
                process.join()
            if self._thread_reader is not None:
                self._thread_reader.join()
        except:
            pass