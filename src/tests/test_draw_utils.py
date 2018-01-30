import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from draw_utils import *
from utils import *

f_path = root + 'test_images/test1.jpg'
img = mpimg.imread(f_path)

# Select a small fraction of pixels to plot by subsampling it
scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
img_small =\
    cv2.resize(img,
               (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
               interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
img_small_RGB = img_small
img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

# Plot and show
plot3d(img_small_RGB, img_small_rgb)
plt.show()

plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
plt.show()

