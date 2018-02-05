import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from features import *
from constants import *


def test_color_hist(f_path):
    image = mpimg.imread(f_path)
    feature_vec, rh, gh, bh, bincen =\
        color_hist(image, nbins=32, bins_range=(0, 256), vis=True)

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
    bin_features = bin_spatial(image)
    # print(bin_features.shape)
    np.set_printoptions(threshold=np.inf)
    print(str(bin_features))
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
    features, hog_image = hog_features2D(gray, orient,
                                         pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=False)
    # print(image.shape, features.shape, hog_image.shape)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()


def test_single_img_features(f_path):
    image = mpimg.imread(f_path)
    assert np.all(bin_spatial(image) ==
                  single_img_features(image,
                  spatial_feat=True,
                  hist_feat=False,
                  hog_feat=False))

    assert np.all(color_hist(image) ==
                  single_img_features(image,
                  spatial_feat=False,
                  hist_feat=True,
                  hog_feat=False))

    assert np.all(hog_features2D(image[:, :, 0]) ==
                  single_img_features(
                  image,
                  spatial_feat=False,
                  hist_feat=False,
                  hog_feat=True))


def test_extract_features(paths):
    image = mpimg.imread(paths[0])
    if paths[0].endswith('.png'):
        image *= 255

    assert np.all(bin_spatial(image) ==
                  extract_features(paths,
                  spatial_feat=True,
                  hist_feat=False,
                  hog_feat=False)[0])

    assert np.all(color_hist(image) ==
                  extract_features(paths,
                  spatial_feat=False,
                  hist_feat=True,
                  hog_feat=False)[0])

    assert np.all(hog_features2D(image[:, :, 0]) ==
                  extract_features(paths,
                  spatial_feat=False,
                  hist_feat=False,
                  hog_feat=True)[0])


if __name__ == '__main__':
    f_path = root + 'test_images/test1.jpg'
    png_path = root + 'test_images/image0550.png'
    # test_color_hist(f_path)
    # test_bin_spatial(png_path)
    # test_hog_features(f_path)
    # test_single_img_features(f_path)
    test_extract_features([png_path])


