import cv2
import numpy as np
from features import *
from utils import *


def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None),
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    row, col, _ = img.shape
    if x_start_stop[0] is None:
        x_start_stop = (0, x_start_stop[1])
    if x_start_stop[1] is None:
        x_start_stop = (x_start_stop[0], col)
    if y_start_stop[0] is None:
        y_start_stop = (0, y_start_stop[1])
    if y_start_stop[1] is None:
        y_start_stop = (y_start_stop[0], row)
    x_stride = int(xy_window[0] * (1 - xy_overlap[0]))
    y_stride = int(xy_window[1] * (1 - xy_overlap[1]))

    window_list = []
    for y_start in range(y_start_stop[0],
                         y_start_stop[1] - xy_window[0] + 1, y_stride):
        for x_start in range(x_start_stop[0],
                             x_start_stop[1] - xy_window[1] + 1, x_stride):
            box = ((x_start, y_start),
                   (x_start + xy_window[0], y_start + xy_window[1]))
            window_list.append(box)
    return window_list


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """
    Define a function you will pass an image
    and the list of windows to be searched (output of slide_windows())
    """

    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1],
                                  window[0][0]:window[1][0]], (64, 64))
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)

        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


def fast_search_windows(img, clf, X_scaler,
                        ystart, ystop, scale=1, color_space='RGB',
                        spatial_size=(32, 32), hist_bins=32,
                        orient=9, pix_per_cell=8, cell_per_block=2,
                        cells_per_step=2):
    """
    Define a single function that can extract features using
    hog sub-sampling and make predictions

    return hog_windows instead the image
    """
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cvtColor(img_tosearch, color_space=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1] / scale),
                                      np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    # nfeat_per_block = orient * cell_per_block**2

    window_size = 64
    nblocks_per_window = (window_size // pix_per_cell) - cell_per_block + 1
    # Instead of overlap, define how many cells to step
    # the move from one block to next block is one cell step
    # nxsteps, nysteps index the size of windows
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    # Compute individual channel HOG features for the entire image
    hog1 = hog_features2D(ch1, orient, pix_per_cell, cell_per_block,
                          feature_vec=False)
    hog2 = hog_features2D(ch2, orient, pix_per_cell, cell_per_block,
                          feature_vec=False)
    hog3 = hog_features2D(ch3, orient, pix_per_cell, cell_per_block,
                          feature_vec=False)

    on_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window_size,
                                                xleft:xleft + window_size],
                                (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features,
                           hist_features,
                           hog_features)).reshape(1, -1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window_size * scale)
                on_windows.append([(xbox_left, ytop_draw + ystart),
                                   (xbox_left + win_draw,
                                    ytop_draw + win_draw + ystart)])

    return on_windows
