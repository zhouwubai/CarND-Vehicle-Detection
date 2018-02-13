import numpy as np
import collections

from utils import (
    add_heat, apply_threshold,
    draw_labeled_bboxes, draw_boxes
)

from scipy.ndimage.measurements import label


class CarDectorPiper(object):

    def __init__(self, detector, keep_n=6,
                 threshold=10, draw_heatmap=True,
                 cthreshold=25, standalone=True):
        self.detector = detector
        self.windows = collections.deque(maxlen=keep_n)
        self.threshold = threshold
        self.cthreshold = cthreshold
        self.draw_heatmap = draw_heatmap
        self.standalone = standalone

    def draw_heatmap_labels(self, shape):
        all_windows = []
        for w in self.windows:
            all_windows.extend(w)

        heatmap = np.zeros(shape)
        heatmap = add_heat(heatmap, all_windows)
        heatmap = apply_threshold(heatmap, self.threshold)
        labels = label(heatmap)
        return labels, heatmap

    def process(self, image):
        draw_image = np.zeros_like(image)
        y_start_stop = [(350, 500), (400, 550), (350, 650)]
        scale = [1, 2, 3]
        cells_per_step = [1, 1, 1]

        hot_windows =\
            self.detector.search_windows(image,
                                         y_start_stop=y_start_stop,
                                         scale=scale,
                                         cells_per_step=cells_per_step)
        self.windows.append(hot_windows)

        alive_box = hot_windows
        if self.draw_heatmap:
            labels, heatmap = self.draw_heatmap_labels(draw_image.shape[:2])
            window_img, alive_box =\
                draw_labeled_bboxes(draw_image, labels,
                                    heatmap, self.cthreshold)
        else:
            window_img = draw_boxes(draw_image, hot_windows,
                                    color=(0, 0, 255), thick=6)
        if self.standalone:
            return window_img
        else:
            return alive_box

