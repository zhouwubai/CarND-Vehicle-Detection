import numpy as np
import cv2
from skimage.feature import hog
from utils import cvtColor
import matplotlib.image as mpimg


def color_hist(img, nbins=32, bins_range=(0, 256)):
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0: len(bin_edges) - 1]) / 2
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return rhist, ghist, bhist, bin_centers, hist_features


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    feature_image = cvtColor(img, color_space=color_space)
    feature_image = cv2.resize(feature_image, size)
    features = feature_image.ravel()

    return features


def hog_features(img, orient, pix_per_cell, cell_per_block,
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


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    features = []
    for img in imgs:
        image = mpimg.imread(img)
        feature_image = cvtColor(image, color_space=color_space)
        spatial_feature_vec = bin_spatial(feature_image, size=spatial_size)
        hist_feature_vec = color_hist(feature_image,
                                      nbins=hist_bins,
                                      bins_range=hist_range)
        features.append(np.concatenate((spatial_feature_vec,
                                        hist_feature_vec)))
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
        img_features.append(spatial_features)
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(hog_features(feature_image[:, :, channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = hog_features(feature_image[:, :, hog_channel],
                                        orient,
                                        pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True)
        img_features.append(hog_features)

    return np.concatenate(img_features)
