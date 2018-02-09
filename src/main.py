import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import pickle

from search import *
from features import *
from utils import *
from constants import *
from models import *


from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
# from sklearn.cross_validation import train_test_split


# image shape (64, 64, 3)
# TODO: Tweak these parameters and see how the results change.
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
name = ModelType.SVC
load_model = True

if not load_model:
    print('Training model...')
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

    y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
    X_files = cars + notcars

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train_files, X_test_files, y_train, y_test = train_test_split(
        X_files, y, test_size=0.2, random_state=rand_state)

    detector = CarDetector(model_type=name,
                           color_space=color_space,
                           spatial_feat=spatial_feat,
                           spatial_size=spatial_size,
                           hist_feat=hist_feat,
                           hist_bins=hist_bins,
                           hog_feat=hog_feat,
                           orient=orient,
                           pix_per_cell=pix_per_cell,
                           cell_per_block=cell_per_block,
                           hog_channel=hog_channel)
    print(detector.print_params())
    X_train = detector.get_features(X_train_files)
    X_test = detector.get_features(X_test_files)
    t = time.time()
    detector.train(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2),
          'Seconds to train {}...'.format(detector.model_type))

    # Check the score of the classifier
    score = detector.evaluate(X_test, y_test)
    print('Test Accuracy of {} = {}'.format(detector.model_type,
                                            round(score, 4)))
    # Check the prediction time for a single sample
    pickle.dump(detector, open('car_model_{}.pkl'.format(name), 'wb'))

if load_model:
    print('Loading model...')
    detector = pickle.load(open('car_model_{}.pkl'.format(name), 'rb'))
    print(detector.print_params())

print('Start testing')
test_img = root + 'test_images/test6.jpg'
image = mpimg.imread(test_img)
draw_image = np.copy(image)

#
# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

ystart = 300
ystop = 680
scale = [1, 2, 3]
cells_per_step = [1, 1, 1]

t = time.time()
hot_windows = detector.search_windows(image, ystart=ystart, ystop=ystop,
                                      scale=scale,
                                      cells_per_step=cells_per_step)
t2 = time.time()
print(round(t2 - t, 2),
      'Seconds to search and identify {} windows'.format(len(hot_windows)))
draw_heatmap = True
if draw_heatmap:
    heatmap = np.zeros(draw_image.shape[:2])
    heatmap = add_heat(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, 1.5)
    labels = label(heatmap)
    window_img = draw_labeled_bboxes(draw_image, labels)
else:
    window_img = draw_boxes(draw_image, hot_windows,
                            color=(0, 0, 255), thick=6)

plt.imshow(window_img)
plt.savefig('../test_images/test6_labeled.jpg')
# plt.show()
