import numpy as np
import cv2
from skimage.feature import hog
from utils import cvtColor
import matplotlib.image as mpimg


def color_hist(img, nbins=32, bins_range=(0, 256), vis=False):
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0: len(bin_edges) - 1]) / 2
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    if vis:
        return hist_features, rhist, ghist, bhist, bin_centers
    else:
        return hist_features


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    feature_image = cvtColor(img, color_space=color_space)
    feature_image = cv2.resize(feature_image, size)
    features = feature_image.ravel()

    return features


def hog_features2D(img, orient=9, pix_per_cell=8, cell_per_block=2,
                   vis=False, feature_vec=True):
    if vis:
        # use skimage.hog() to get both features and a visualization
        features, hog_image =\
            hog(img, orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=vis, feature_vector=feature_vec)
        return features


def hog_features3D(img, hog_channel, orient,
                   pix_per_cell, cell_per_block):
    features = []
    if hog_channel == 'ALL':
        for channel in range(img.shape[2]):
            features.extend(hog_features2D(img[:, :, channel],
                            orient, pix_per_cell, cell_per_block,
                            vis=False, feature_vec=True))
    else:
        features = hog_features2D(img[:, :, hog_channel],
                                  orient,
                                  pix_per_cell, cell_per_block,
                                  vis=False, feature_vec=True)
    return features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for img in imgs:
        # to handle difference scale for different image formta
        image = mpimg.imread(img)
        if img.endswith('.png'):
            image *= 255
            image = image.astype(np.uint8)

        feature = single_img_features(image, color_space, spatial_size,
                                      hist_bins, orient, pix_per_cell,
                                      cell_per_block, hog_channel,
                                      spatial_feat, hist_feat, hog_feat)
        features.append(feature)
    return features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Define a function to extract features from a single image window
    This function is very similar to extract_features()
    just for a single image rather than list of images
    """
    img_features = []
    feature_image = cvtColor(img, color_space=color_space)
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # print('spatial:', spatial_features)
        img_features.append(spatial_features)
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # print('hist:', hist_features)
        img_features.append(hist_features)
    if hog_feat:
        hog_feature = hog_features3D(feature_image, hog_channel, orient,
                                     pix_per_cell, cell_per_block)
        # print('hog:', hog_feature)
        img_features.append(hog_feature)

    img_features = np.concatenate(img_features)
    # print(img_features)
    return img_features
