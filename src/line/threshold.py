import numpy as np
import cv2


class BaseThreshold(object):
    def applyThreshold(self, image, thresh):
        raise NotImplementedError


class GradientAbsThreshold(BaseThreshold):
    def __init__(self, orient, sobel_kernel=3):
        super(GradientAbsThreshold, self).__init__()
        self.orient = orient
        self.sobel_kernel = sobel_kernel

    def applyThreshold(self, image, thresh):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) &
                      (scaled_sobel <= thresh[1])] = 1
        return binary_output


class GradientMagThreshold(BaseThreshold):
    def __init__(self, sobel_kernel=3):
        super(GradientMagThreshold, self).__init__()
        self.sobel_kernel = sobel_kernel

    def applyThreshold(self, image, thresh):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        sobelxy = np.sqrt(sobelx**2 + sobely**2)
        scaled_sobel = np.uint8(255 * sobelxy / np.max(sobelxy))

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) &
                      (scaled_sobel <= thresh[1])] = 1
        return binary_output


class GradientDirThreshold(BaseThreshold):
    def __init__(self, sobel_kernel=15):
        super(GradientDirThreshold, self).__init__()
        self.sobel_kernel = sobel_kernel

    def applyThreshold(self, image, thresh):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx) + 1e-6
        abs_sobely = np.absolute(sobely)
        scaled_sobel = np.arctan(abs_sobely / abs_sobelx)

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) &
                      (scaled_sobel <= thresh[1])] = 1
        return binary_output


class ColorThreshold(BaseThreshold):
    def __init__(self, cmap='gray', channel=0):
        super(ColorThreshold, self).__init__()
        self.cmap = cmap
        self.channel = channel

    def cvtColor(self, image):
        if self.cmap == 'RGB':
            return image[:, :, self.channel]
        elif self.cmap == 'HLS':
            HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            return HLS[:, :, self.channel]
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def applyThreshold(self, image, thresh):
        cvt_image = self.cvtColor(image)

        binary_output = np.zeros_like(cvt_image)
        binary_output[(cvt_image > thresh[0]) & (cvt_image <= thresh[1])] = 1
        return binary_output


THRESHOLDS = {
    'gray': [180, 255],
    'R': [190, 255],
    'G': [170, 255],
    'B': [0, 100],
    'H': [15, 100],
    'L': [150, 255],
    'S': [90, 255],
    'X': [10, 150],
    'Y': [10, 150],
    'XY': [20, 200],
    'X/Y': [0.7, 1.3]
}

VOTERS = {
    'gray': ColorThreshold('gray'),
    'R': ColorThreshold('RGB', 0),
    'G': ColorThreshold('RGB', 1),
    'B': ColorThreshold('RGB', 2),
    'H': ColorThreshold('HLS', 0),
    'L': ColorThreshold('HLS', 1),
    'S': ColorThreshold('HLS', 2),
    'X': GradientAbsThreshold('x', sobel_kernel=3),
    'Y': GradientAbsThreshold('y', sobel_kernel=3),
    'XY': GradientMagThreshold(sobel_kernel=3),
    'X/Y': GradientDirThreshold(sobel_kernel=15)
}


def majority_vote(image, thresh_names, n_vote):
    global THRESHOLDS
    global VOTERS

    shape = image.shape
    sum_binary = np.zeros(shape[:2])
    for name in thresh_names:
        sum_binary += VOTERS[name].applyThreshold(image, THRESHOLDS[name])

    vote_binary = np.zeros_like(sum_binary)
    vote_binary[sum_binary >= n_vote] = 1

    return vote_binary


