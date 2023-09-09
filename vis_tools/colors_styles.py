import numpy as np


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.link_pairs)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))


color2 = [(252, 176, 243), (252, 176, 243), (252, 176, 243),
          (0, 176, 240), (0, 176, 240), (0, 176, 240),
          (255, 255, 0), (255, 255, 0), (169, 209, 142),
          (169, 209, 142), (169, 209, 142),
          (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127)]

link_pairs2 = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [9, 7], [7, 5], [5, 6], [6, 8], [8, 10],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
]

point_color2 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (252, 176, 243), (0, 176, 240), (252, 176, 243),
                (0, 176, 240), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]
