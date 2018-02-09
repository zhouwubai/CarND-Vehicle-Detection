import numpy as np
import pickle

from search import *
from features import *
from utils import *
from constants import *
from models import *

from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


# image shape (64, 64, 3)
color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat = True  # Spatial features on or off, None < 0, 768, 0.89
spatial_size = (16, 16)  # Spatial binning dimensions
hist_feat = True  # Histogram features on or off, 48, 0.9654
hist_bins = 16  # Number of histogram bins
hog_feat = True  # HOG features on or off, 1728, 0.885
orient = 16   # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
name = ModelType.DecisionTree

print('Loading model...')
detector = pickle.load(open('car_model_{}.pkl'.format(name), 'rb'))
print(detector.print_params())


def pipeline(image, detector=detector):
    draw_image = np.copy(image)
    y_start_stop = [(350, 500), (400, 550), (350, 650)]
    scale = [1, 2, 3]
    cells_per_step = [1, 1, 1]

    hot_windows = detector.search_windows(image, y_start_stop=y_start_stop,
                                          scale=scale,
                                          cells_per_step=cells_per_step)
    heatmap = np.zeros(draw_image.shape[:2])
    heatmap = add_heat(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, 2)
    labels = label(heatmap)
    window_img = draw_labeled_bboxes(draw_image, labels)
    return window_img


white_output = '../project_video_labeled.mp4'
clip1 = VideoFileClip("../project_video.mp4")
white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(white_output, audio=False)


