import numpy as np

from utils import (
    eval_poly,
    curvature_poly
)


class ImbalanceCoordinateNum(Exception):
    pass


class Line(object):
    """
    Class definition for line

    Parameters
    -----------
    allx: array of int
        detected x coordinate values
    ally: array of int
        detected y coordinate values
    minpix: int
        the minimum dots needed to fit a sufficient line
    ym_per_pix: float
        unit transform from pix to meters in y axis
    xm_per_pix:float
        unit transform from pix to meters in x axis
    image_size: tuple of float
        the size of image the line was detected
    """
    def __init__(self, minpix=50,
                 ym_per_pix=30.0 / 720,
                 xm_per_pix=3.7 / 700,
                 image_size=(1280, 720)):

        self.allx = []
        self.ally = []
        self.minpix = minpix
        self.image_size = image_size
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix

        # polynomial coefficient for the fit
        self.coeffs = None
        # radius of curvature of the line in meters
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.base_pos = None

    def fit(self, allx, ally):
        if len(allx) != len(ally):
            raise ImbalanceCoordinateNum()

        self.allx = allx
        self.ally = ally

        if self.is_sufficient():
            self.coeffs = np.polyfit(ally, allx, 2)
            self._update()

    def set_coeffs(self, coeffs):
        self.coeffs = coeffs
        self._update()

    def _update(self):
        self.curvature()
        self.distance_from_center()

    def curvature(self):
        y = self.image_size[1]
        new_a = self.coeffs[0] * self.xm_per_pix / (self.ym_per_pix ** 2)
        new_b = self.coeffs[1] * self.xm_per_pix / self.ym_per_pix
        new_y = y * self.ym_per_pix
        self.radius_of_curvature = curvature_poly([new_a, new_b, 0], new_y)

    def distance_from_center(self):
        """
        positve: right
        negative: left
        """
        mid_x, max_y = self.image_size[0] / 2.0, self.image_size[1]
        bottom_x = eval_poly(self.coeffs, max_y)
        self.base_pos = (bottom_x - mid_x) * self.xm_per_pix

    def is_sufficient(self):
        return (len(self.allx) >= self.minpix or self.coeffs is not None)

    def difference(self, other):
        return self.fit - other.fit

    def is_parallel(self, other, max_error=1.0):
        if self.is_sufficient() or other.is_sufficient():
            return False
        else:
            diffs = np.absolute(self.difference(other)[: 2])
            # print("X1 - X2: {}, sum: {}".format(diffs, np.sum(diffs)))
            return (np.sum(diffs) < max_error)

    # should check the consistency of those lines, such as image_size
    @classmethod
    def calculate_width(cls, left, right):
        return right.base_pos - left.base_pos

    @classmethod
    def deviate_of_center(cls, left, right):
        x, y = left.image_size[0], left.image_size[1]
        x_left = eval_poly(left.coeffs, y)
        x_right = eval_poly(right.coeffs, y)
        return (x / 2.0 - (x_left + x_right) / 2.0) * left.xm_per_pix

