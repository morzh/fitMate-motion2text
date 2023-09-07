from copy import deepcopy
import random
import numpy as np

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class Interpolation:
    def __init__(self, n_filter_window_length: int = 5):
        self.n_filter_window_length = n_filter_window_length

    def interpolate_skeletons(self, skeletons: list):
        x, x_none, skeletons = self.get_interpolation_data(skeletons)
        np_skeletons = np.array(skeletons)
        n_axis = np_skeletons.shape[-1]
        new_points = [[] for _ in range(n_axis)]

        for ax_idx in range(n_axis):
            for vector in np_skeletons[..., ax_idx].swapaxes(0, 1):
                vector = savgol_filter(vector, self.n_filter_window_length, 2)
                f = interp1d(x, vector, kind='cubic')
                full_vector = f(x_none)
                full_vector = savgol_filter(full_vector, self.n_filter_window_length * 2 + 1, 2)
                new_points[ax_idx].append(full_vector)
        new_skeletons = np.array(new_points).swapaxes(0, -1)

        for new_skeleton_idx, skeleton in zip(x_none, new_skeletons):
            skeletons.insert(new_skeleton_idx, skeleton)
        return skeletons

    def stretch_motion_bert(self, skeletons: np.ndarray, k: float, smoothing: bool = False):
        n_axis = skeletons.shape[-1]
        new_length = round((skeletons.shape[1]) * k)

        old_time_positions = np.arange(skeletons.shape[1])
        new_time_positions = np.arange(new_length) / k
        new_time_positions[-1] = old_time_positions[-1]

        shape = list(skeletons.shape)
        shape[1] = new_length
        new_points = np.zeros(shape)

        for i, dim in enumerate(skeletons):
            for ax_idx in range(n_axis):
                for j, vector in enumerate(dim[..., ax_idx].swapaxes(0, 1)):
                    if smoothing:
                        vector = savgol_filter(vector, self.n_filter_window_length, 2)
                    f = interp1d(old_time_positions, vector, kind='cubic')
                    full_vector = f(new_time_positions)
                    if smoothing:
                        full_vector = savgol_filter(
                            full_vector, self.n_filter_window_length * 2 + 1, 2
                        )
                    new_points[i, :, j, ax_idx] = full_vector
        return new_points

    @staticmethod
    def get_interpolation_data(skeletons: list) -> (list, list, list):
        clean_skeletons = []
        x = []
        x_none = []
        for i, sk in enumerate(skeletons):
            if sk is None:
                x_none.append(i)
            else:
                x.append(i)
                clean_skeletons.append(sk)
        return x, x_none, clean_skeletons

    @staticmethod
    def threshold_move_distance(skeletons: list) -> list:
        # doesn't work
        skeletons_dif = (np.array(skeletons)[:-1] - np.array(skeletons)[1:])
        skeletons_dif[..., :2][skeletons_dif[..., :2] > 0.01] = 0.01
        skeletons_dif[..., :2][skeletons_dif[..., :2] < -0.01] = -0.01
        skeletons_dif[..., 2][skeletons_dif[..., 2] > 0.1] = 0.1
        skeletons_dif[..., 2][skeletons_dif[..., 2] < -0.1] = -0.1
        new_skeletons = [skeletons[0]]
        for sk_dif in skeletons_dif:
            new_skeletons.append(new_skeletons[-1] + sk_dif)
        return new_skeletons

    @staticmethod
    def drop_percent_of_skeletons(skeletons: list, drop_percent_frames: float = 0.2) -> list:
        skeletons = deepcopy(skeletons)
        length = len(skeletons)
        drop_indexes = [i for i in range(2, length - 2)]
        random.shuffle(drop_indexes)
        drop_indexes = drop_indexes[:int(length * drop_percent_frames)]
        for idx in drop_indexes:
            skeletons[idx] = None
        return skeletons

    @staticmethod
    def save_each_n_skeletons(skeletons: list) -> list:
        new_skeletons = [skeletons[0]]
        n_take_only_each = 4

        empty_list = [None for _ in range(n_take_only_each - 1)]
        for sk in skeletons[1::n_take_only_each]:
            new_skeletons.extend(empty_list)
            new_skeletons.append(sk)
        return new_skeletons
