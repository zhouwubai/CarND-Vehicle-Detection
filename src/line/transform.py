import numpy as np
import cv2
import pickle

import matplotlib.image as mpimg


class Calibrator(object):

    def __init__(self, mtx=None, dist=None, size=(9, 4), path=None):
        self.mtx = mtx
        self.dist = dist
        if path is not None:
            self.load_params(path)

    def load_params(self, path):
        self.mtx, self.dist = pickle.load(open(path, 'rb'))

    def save_params(self, path):
        pickle.dump((self.mtx, self.dist),
                    open(path, 'wb'))

    def get_params(self):
        return self.mtx, self.dist

    def fit(self, files):
        """
        Using a list of chessboard files to calibrate the camera

        Parameters
        files: list of string
            the file path of images used to do calibration
        size: tuple (M, N):
            the size of chess board
        savefile: string
            file to save the fitted parameters

        Return
        mtx: matrix
            intrisic camera parameters fitted
        dist: vector
            distortion coefficent fitted
        """
        size = self.size
        # destination coordinates. The third dim is a virtual dimension
        objp = np.zeros((size[1] * size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images
        objpoints = []  # 3d objects in real world space
        imgpoints = []  # 2d points in image plane.

        img_size = None
        for idx, fname in enumerate(files):
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # find the chessboard color
            ret, corners = cv2.findChessboardCorners(gray, size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

            if not img_size:
                img_size = (img.shape[1], img.shape[0])

        ret, mtx, dist, rvecs, tvecs =\
            cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        self.mtx, self.dist = mtx, dist


class Unwarper(object):

    def __init__(self, M=None, inv_M=None, path=None):
        self.M = M
        self.inv_M = inv_M
        if path is not None:
            self.load_params(path)

    def load_params(self, path):
        self.M, self.inv_M = pickle.load(open(path, 'rb'))

    def save_params(self, path):
        pickle.dump((self.M, self.inv_M),
                    open(path, 'wb'))

    def get_params(self):
        return self.M, self.inv_M

    def fit(self, src, dst):
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.inv_M = cv2.getPerspectiveTransform(dst, src)

    @classmethod
    def fit_chessboards(cls, undist, offset=100, size=(9, 6)):
            nx = size[0]
            gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
            img_size = (gray.shape[1], gray.shape[0])
            ret, corners = cv2.findChessboardCorners(gray, size, None)

            M, inv_M = None, None
            if ret:
                # top_left, top_right, bottom_right, bottom_left
                src = np.float32([corners[0], corners[nx - 1],
                                 corners[-1], corners[-nx]])
                dst = np.float32([[offset, offset],
                                  [img_size[0] - offset, offset],
                                  [img_size[0] - offset, img_size[1] - offset],
                                  [offset, img_size[1] - offset]])
                M = cv2.getPerspectiveTransform(src, dst)
                inv_M = cv2.getPerspectiveTransform(dst, src)
            return ret, M, inv_M


