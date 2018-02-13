import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D


def cvtColor(img, color_space='RGB'):
    """
    Convert RGB image to other color_space
    """
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    return feature_image


def add_heat(heatmap, bbox_list, value=1):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += value

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels, heatmap=None, threshold=25):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        if heatmap is not None:
            max_val = np.max(heatmap[bbox[0][1]:bbox[1][1],
                                     bbox[0][0]:bbox[1][0]])
            if max_val >= threshold:
                cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        else:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def eval_poly(coeffs, y):
    return (coeffs[0] * y ** 2 + coeffs[1] * y + coeffs[2])


def curvature_poly(coeffs, y):
    """
    calculate the curvature of a polyline at point y
    """
    a, b = coeffs[0], coeffs[1]
    return ((1 + (2 * a * y + b) ** 2) ** 1.5 / np.absolute(2 * a))


# drawing utils
def drawChessboardCorners(file, size=(9, 6), savefile=None):
    img = mpimg.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # find the chessboard corner
    ret, corners = cv2.findChessboardCorners(gray, size, None)
    cv2.drawChessboardCorners(img, size, corners, ret)
    if ret:
        plt.imshow(img)
        if savefile:
            plt.savefig(savefile)
        plt.show()
    else:
        print("Corner not find")


def drawUndistortedImage(original, undistorted, savefile=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(original)
    ax1.set_title('Originnal Image', fontsize=30)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=30)
    if savefile:
        plt.savefig(savefile)
    plt.show()


def drawUnwarpedImage(undistorted, unwarped, savefile=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(undistorted)
    ax1.set_title('Undistorted Image', fontsize=30)
    ax2.imshow(unwarped)
    ax1.set_title('Unwarped Image', fontsize=30)
    if savefile:
        plt.savefig(savefile)
    plt.show()


def drawImage(images, savefile=None):
    """
    Draw binary images, assuming N%3 = 0
    """
    N = len(images)
    row = int(N / 3)
    fig, axis = plt.subplots(row, 3, figsize=(20, 4 * row))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axis = axis.ravel()
    for i in range(N):
        axis[i].imshow(images[i])
    if savefile:
        plt.savefig(savefile)
    plt.tight_layout()
    plt.show()


def drawBinaryImage(images, n_col=3, title=None, savefile=None):
    """
    Draw binary images, assuming N%3 = 0
    """
    if title:
        assert len(images) == len(title)
    N = len(images)
    row = int(N / n_col)
    fig, axis = plt.subplots(row, n_col, figsize=(7 * n_col, 4 * row))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axis = axis.ravel()
    for i in range(N):
        axis[i].imshow(images[i], cmap='gray')
        if title:
            axis[i].set_title(title[i])
            # axis[i].set_axis_off()
    if savefile:
        plt.savefig(savefile)
    plt.tight_layout()
    plt.show()


def save_figures(images, filenames):
    assert len(images) == len(filenames)
    for image, filename in zip(images, filenames):
        cv2.imwrite(filename, np.dstack((image, image, image)) * 255)


def draw_lines(left_fit, right_fit, ymax=720):
    ploty = np.linspace(0, ymax - 1, ymax)
    l_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    r_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    plt.plot(l_fitx, ploty, color='yellow')
    plt.plot(r_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, inplace=False):
    if inplace:
        draw_img = img
    else:
        draw_img = np.copy(img)
    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img


def plot3d(pixels, colors_rgb, axis_labels=list("RGB"),
           axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


if __name__ == '__main__':
    f_path = '../test_images/test1.jpg'
    image = mpimg.imread(f_path)

    bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]
    result = draw_boxes(image, bboxes)
    plt.imshow(result)
    plt.show()



