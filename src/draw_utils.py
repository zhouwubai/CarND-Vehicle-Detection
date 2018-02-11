import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


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
    row = int(N/3)
    fig, axis = plt.subplots(row, 3, figsize=(20, 4*row))
    fig.subplots_adjust(hspace = .2, wspace=.001)
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
    row = int(N/n_col)
    fig, axis = plt.subplots(row, n_col, figsize=(7*n_col, 4*row))
    fig.subplots_adjust(hspace = .2, wspace=.001)
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


def saveFigures(images, filenames):
    assert len(images) == len(filenames)
    for image, filename in zip(images, filenames):
        cv2.imwrite(filename, np.dstack((image, image, image)) * 255)

def drawLines(left_fit, right_fit, ymax=720):
    ploty = np.linspace(0, ymax-1, ymax)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()