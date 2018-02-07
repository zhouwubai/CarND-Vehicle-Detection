import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
from sklearn.preprocessing import StandardScaler

from search import *
from features import *
from utils import *
from constants import *
from models import *

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

# Read in cars and notcars
data_root = root + 'data/'
images = glob.glob(data_root + '**/*.png', recursive=True)
cars = []
notcars = []
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
# uncomment

"""
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]
"""

# image shape (64, 64, 3)
# TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat = False  # Spatial features on or off, None < 0, 768, 0.89
spatial_size = (16, 16)  # Spatial binning dimensions

hist_feat = False  # Histogram features on or off, 48, 0.9654
hist_bins = 16  # Number of histogram bins

hog_feat = True  # HOG features on or off, 2548, 0.885
orient = 16   # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"


car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel,
                                spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size,
                                   hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel,
                                   spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
print(X.shape)
# X = X.astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))


name = ModelType.DecisionTree
clf = get_classifier(name=name)
# Check the training time for the training
t = time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train {}...'.format(name))
# Check the score of the classifier
print('Test Accuracy of {} = '.format(name),
      round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

test_img = root + 'test_images/test1.jpg'
image = mpimg.imread(test_img)
draw_image = np.copy(image)

#
# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255
x_start_stop = [None, None]  # Min and max in x to search in slide_window()
y_start_stop = [None, None]  # Min and max in y to search in slide_window()

windows = slide_window(image, x_start_stop=x_start_stop,
                       y_start_stop=y_start_stop,
                       xy_window=(128, 128), xy_overlap=(0.6, 0.6))
print(len(windows))
hot_windows = search_windows(image, windows, clf, X_scaler,
                             color_space=color_space,
                             spatial_size=spatial_size, hist_bins=hist_bins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel,
                             spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

# plt.imshow(window_img)
# plt.show()
