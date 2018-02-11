import copy
import collections
import numpy as np
import cv2

from line.search import (
    find_lines,
    find_lines_with_previous,
    curvature_pixel_2_meter,
    evaluate,
    deviate_of_center
)


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, fit, fitx, fity,
                 ym_per_pix=28 / 720, xm_per_pix=3.7 / 650,
                 image_size=(1280, 720)):
        # polynomial coefficients for the most recent fit
        self.fit = fit
        self.allx = fitx
        self.ally = fity
        self.image_size = image_size
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix

        self.radius_of_curvature = None
        self.base_pos = None
        self.average_x = None
        self.update()

    def update(self):
        """
        update line after some changes
        """
        self.update_curvature()
        self.update_base_pose()

    def shift(self, distance):
        """
        shift a line by number of meter, negative->left, postive->right
        """
        self.fit[2] += distance / self.xm_per_pix
        self.update()

    def is_empty(self):
        return (self.fit is None)

    def num_points(self):
        return len(self.allx)

    def update_curvature(self):
        if not self.is_empty():
            self.radius_of_curvature =\
                curvature_pixel_2_meter(self.fit, self.image_size[1],
                                        self.ym_per_pix, self.xm_per_pix)

    def update_base_pose(self):
        if not self.is_empty():
            bottom_x = evaluate(self.fit, self.image_size[0])
            self.base_pos =\
                (bottom_x - self.image_size[1] / 2) * self.xm_per_pix

    def is_parallel(self, other, max_error=1.0):
        if self.is_empty() or other.is_empty():
            return False
        else:
            diffs = np.absolute(self.difference(other)[: 2])
            # print("X1 - X2: {}, sum: {}".format(diffs, np.sum(diffs)))
            return (np.sum(diffs) < max_error)

    def difference(self, other):
        return self.fit - other.fit


class Lane():
    def __init__(self, left_line=None, right_line=None):
        self.left_line = left_line
        self.right_line = right_line
        self.lane_width = None
        self.update()

    def update(self):
        self.update_width()

    def update_width(self):
        if not self.left_line.is_empty() and not self.right_line.is_empty():
            self.lane_width =\
                self.right_line.base_pos - self.left_line.base_pos

    def get_raw_lines(self):
        return (self.left_line.fit, self.right_line.fit)


class Road():
    def __init__(self, keep_n=10, alpha=0.8,
                 ym_per_pix=28 / 720, xm_per_pix=3.7 / 650):

        self.alpha = alpha
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix

        self.detected = False
        self.history = collections.deque(maxlen=keep_n)
        self.images = []
        self.warped = []
        self.results = []

        self.lanes_before = []
        self.lanes_after = []

    def is_history_ready(self):
        """
        in case history has no data
        """
        return len(self.history) > 0

    def peek_history_last(self):
        return self.history[-1]

    def best_left_fit(self):
        return np.average([_.left_line.fit for _ in self.history], axis=0)

    def best_right_fit(self):
        return np.average([_.right_line.fit for _ in self.history], axis=0)

    def best_lane_width(self):
        return np.average([_.lane_width for _ in self.history])

    def smooth_lane(self, lane):
        if lane.left_line.is_empty() and lane.right_line.is_empty():
            lane.left_line = copy.deepcopy(self.history[-1].left_line)
            lane.right_line = copy.deepcopy(self.history[-1].right_line)
            lane.update()
            return
        elif lane.left_line.is_empty():
            lane.left_line = copy.deepcopy(self.history[-1].left_line)
        elif lane.right_line.is_empty():
            lane.right_line = copy.deepcopy(self.history[-1].right_line)

        # smooth with history here
        lane.left_line.fit =\
            (1 - self.alpha) * self.best_left_fit() +\
            self.alpha * lane.left_line.fit
        lane.left_line.update()
        lane.right_line.fit =\
            (1 - self.alpha) * self.best_right_fit() +\
            self.alpha * lane.right_line.fit
        lane.right_line.update()
        lane.update()

    def is_high_quality(self, lane, error=0.8):
        if lane.left_line.fit is None or lane.right_line.fit is None:
            return False

        flag1 = lane.left_line.is_parallel(lane.right_line)
        diff = np.absolute(lane.lane_width - self.best_lane_width())
        # print(lane.lane_width, self.best_lane_width(), diff)
        flag2 = (diff > error)
        return (flag1 and flag2)

    def sanityCheck(self, left_fit, right_fit,
                    leftx, lefty, rightx, righty, refit=False):
        if not refit:
            if left_fit is None and right_fit is None:
                return (False, left_fit, right_fit)

        left_line = Line(left_fit, leftx, lefty)
        right_line = Line(right_fit, rightx, righty)
        new_lane = Lane(left_line, right_line)

        self.lanes_before.append(copy.deepcopy(new_lane))
        # not enought history, just add them
        if not self.is_history_ready():
            self.history.append(new_lane)
        else:
            # one append high quality lane to history
            if self.is_high_quality(new_lane):
                self.detected = True
                self.history.append(new_lane)
            else:
                self.detected = False
                self.smooth_lane(new_lane)
        self.lanes_after.append(new_lane)

        return (True, new_lane.left_line.fit, new_lane.right_line.fit)

    def process_image(self, image):
        # step one: undistorted image
        shape = image.shape
        img_size = (shape[1], shape[0])
        undistorted = cv2.undistort(image, mtx, dist, None, mtx)
        unwarped = cv2.warpPerspective(undistorted, M, img_size)
        binary_warped = majority_vote(unwarped, thresh_names, n_vote)

        self.images.append(unwarped)
        self.warped.append(binary_warped)

        # already know line from previous frame
        is_passed = False
        if self.detected:
            hist_left_fit, hist_right_fit =\
                self.peek_history_last().get_raw_lines()
            left_fit, right_fit, leftx, lefty, rightx, righty =\
                find_lines_with_previous(binary_warped,
                                         hist_left_fit, hist_right_fit)
            # sanityCheck
            is_passed, left_fit, right_fit =\
                self.sanityCheck(left_fit, right_fit,
                                 leftx, lefty, rightx, righty, False)

        # detect line from stratch
        if not is_passed:
            left_fit, right_fit, leftx, lefty, rightx, righty =\
                find_lines(binary_warped)
            # sanityCheck
            is_passed, left_fit, right_fit =\
                self.sanityCheck(left_fit, right_fit,
                                 leftx, lefty, rightx, righty, True)

        ploty = np.linspace(0, shape[0] - 1, shape[0])
        left_fitx =\
            left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx =\
            right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                                                ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, invM, img_size)

        # get the curvature and put text
        y_eval = np.max(ploty)
        left_curve = curvature_pixel_2_meter(left_fit, y_eval,
                                             self.ym_per_pix, self.xm_per_pix)
        right_curve = curvature_pixel_2_meter(right_fit, y_eval,
                                              self.ym_per_pix, self.xm_per_pix)

        avg_curve = (left_curve + right_curve) / 2
        curvature_txt = "Radius of Curvature = {:.0f}(m)".format(avg_curve)
        deviation =\
            deviate_of_center(left_fit, right_fit, shape) * self.xm_per_pix
        if deviation >= 0:
            dtxt = "Vechicle is {:.2f}(m) right of center".format(deviation)
        else:
            dtxt = "Vechicle is {:.2f}(m) left of center".format(-deviation)

        cv2.putText(undistorted, curvature_txt, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(undistorted, dtxt, (100, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        self.results.append(result)
        return result


