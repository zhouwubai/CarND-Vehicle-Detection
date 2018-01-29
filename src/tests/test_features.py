import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import features as F
from utils import root


def test_color_hist(f_path):
    image = mpimg.imread(f_path)
    rh, gh, bh, bincen, feature_vec =\
        F.color_hist(image, nbins=32, bins_range=(0, 256))

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
        plt.show()
    else:
        print('Your function is returning None for at least one variable...')


def test_bin_spatial(f_path):
    image = mpimg.imread(f_path)
    bin_features = F.bin_spatial(image)
    plt.plot(bin_features)
    plt.title('Spatially Binned Features')
    plt.show()


def test_hog_features(f_path):
    image = mpimg.imread(f_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # call our functions with vis = True
    features, hog_image = F.hog_features(gray, orient,
                                         pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=False)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()


if __name__ == '__main__':
    f_path = root + 'test_images/test1.jpg'
    # test_color_hist(f_path)
    # test_bin_spatial(f_path)
    test_hog_features(f_path)





