import copy
import collections
import numpy as np
import cv2

from line.base import Line
from line.search import (
    search_lines,
    fast_search_lines,
)
from line.threshold import majority_vote
from utils import draw_boxes


class LineDetector(object):
    def __init__(self, calibrator=None, unwarper=None,
                 keep_n=10, alpha=0.8):

        self.calibrator = calibrator
        self.unwarper = unwarper
        self.alpha = alpha

        self.detected = False
        self.history = collections.deque(maxlen=keep_n)

    def avg_left_coeffs(self):
        return np.average([_[0].coeffs for _ in self.history], axis=0)

    def avg_right_coeffs(self):
        return np.average([_[1].coeffs for _ in self.history], axis=0)

    def best_lane_width(self):
        return np.average([Line.calculate_width(*lines)
                          for lines in self.history])

    def is_high_quality(self, left_line, right_line, error=0.8):
        if not left_line.is_sufficient() and not right_line.is_sufficient():
            return False

        flag1 = left_line.is_parallel(right_line)
        lane_width = Line.calculate_width(left_line, right_line)
        diff = np.absolute(lane_width - self.best_lane_width())
        # print(lane.lane_width, self.best_lane_width(), diff)
        flag2 = (diff > error)
        return (flag1 and flag2)

    def smooth_lines(self, left_line, right_line):
        if (not left_line.is_sufficient() and
                not right_line.is_sufficient()):
            return copy.deepcopy(self.history[-1])
        elif not left_line.is_sufficient():
            left_line = copy.deepcopy(self.history[-1][0])
        elif not right_line.is_sufficient():
            right_line = copy.deepcopy(self.history[-1][1])

        # smooth with history here
        left_line.set_coeffs((1 - self.alpha) * self.avg_left_coeffs() +
                             self.alpha * left_line.coeffs)

        right_line.set_coeffs((1 - self.alpha) * self.avg_right_coeffs() +
                              self.alpha * right_line.coeffs)
        return left_line, right_line

    def sanityCheck(self, left_line, right_line, refit=False):
        if not refit:
            if left_line.is_sufficient() and right_line.is_sufficient():
                return (False, left_line, right_line)

        # not enought history, just add them
        if len(self.history) == 0:
            self.history.append((left_line, right_line))
        else:
            # one append high quality lane to history
            if self.is_high_quality(left_line, right_line):
                self.detected = True
                self.history.append((left_line, right_line))
            else:
                self.detected = False
                left_line, right_line =\
                    self.smooth_lines(left_line, right_line)

        return (True, left_line, right_line)

    def process_image(self, image):
        # step one: undistorted image
        import pdb
        pdb.set_trace()
        thresh_names, n_vote = ['S', 'R', 'X'], 2
        shape = image.shape
        img_size = (shape[1], shape[0])
        undistorted = cv2.undistort(image, self.calibrator.mtx,
                                    self.calibrator.dist, None,
                                    self.calibrator.mtx)

        binary_warped = majority_vote(undistorted, thresh_names, n_vote)
        binary_unwarped =\
            cv2.warpPerspective(binary_warped, self.unwarper.M, img_size)

        # already know line from previous frame
        is_passed = False
        if self.detected:
            hist_left_line, hist_right_line = self.history[-1]
            hist_left_coeffs, hist_right_coeffs =\
                hist_left_line.coeffs, hist_right_line.coeffs
            left_line, right_line =\
                fast_search_lines(binary_unwarped,
                                  hist_left_coeffs, hist_right_coeffs)
            # sanityCheck
            is_passed, left_line, right_line =\
                self.sanityCheck(left_line, right_line, False)

        # detect line from stratch
        left_windows, right_windows = [], []
        if not is_passed:
            left_line, right_line, left_windows, right_windows =\
                search_lines(binary_unwarped, draw_windows=True)
            # sanityCheck
            is_passed, left_line, right_line =\
                self.sanityCheck(left_line, right_line, True)

        left_fit, right_fit = left_line.coeffs, right_line.coeffs
        ploty = np.linspace(0, shape[0] - 1, shape[0])
        left_fitx =\
            left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx =\
            right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        unwarp_zero = np.zeros_like(binary_unwarped).astype(np.uint8)
        color_unwarp = np.dstack((unwarp_zero, unwarp_zero, unwarp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                                                ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_unwarp, np.int_([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_unwarp,
                                      self.unwarper.inv_M, img_size)

        # get the curvature and put text
        left_curve = left_line.radius_of_curvature
        right_curve = right_line.radius_of_curvature

        avg_curve = (left_curve + right_curve) / 2
        curvature_txt = "Radius of Curvature = {:.0f}(m)".format(avg_curve)
        deviation = Line.deviate_of_center(left_line, right_line)
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
        return result

    def merge_drawing(self, undistorted, binary_warped, binary_unwarped,
                      left_line, right_line,
                      left_windows, right_windows, scale=4):
        """
        Drawing all information on the frame
        1. fill the space between detected two lines with color (0, 255, 0)
        2. showing binary warped on the top right
        3. showing unwarped with detected lines and windows
        4. put text info of curvature and deviation from center
        """

        shape = undistorted.shape
        img_size = (shape[1], shape[0])

        # first drawing the road
        left_fit, right_fit = left_line.coeffs, right_line.coeffs
        ploty = np.linspace(0, shape[0] - 1, shape[0])
        left_fitx =\
            left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx =\
            right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # draw line and windows on this
        unwarped_line_win =\
            np.dstack((binary_unwarped, binary_unwarped, binary_unwarped))
        unwarped_line_win = 255 * unwarped_line_win.astype(np.uint8)
        unwarped_line_win[ploty, left_fitx.astype(np.uint8)] = [255, 0, 0]
        unwarped_line_win[ploty, right_fitx.astype(np.uint8)] = [0, 0, 255]
        unwarped_line_win =\
            draw_boxes(unwarped_line_win, left_windows,
                       color=(255, 0, 0), thick=5, inplace=True)
        unwarped_line_win =\
            draw_boxes(unwarped_line_win, right_windows,
                       color=(0, 0, 255), thick=5, inplace=True)

        # stack combine binary_warped and unwarp_line_win together
        # add boarder, resize and put into undistorted top right
        warped = np.dstack((binary_warped, binary_warped, binary_warped))
        warped = 255 * warped.astype(np.uint8)
        panel1 = cv2.copyMakeBorder(warped, 10, 10, 10, 5,
                                    cv2.BORDER_CONSTANT, value=(255, 0, 0))
        panel2 = cv2.copyMakeBorder(unwarped_line_win, 10, 10, 5, 10,
                                    cv2.BORDER_CONSTANT, value=(255, 0, 0))
        info_panel = np.vstack((panel1, panel2))
        info_panel_small = cv2.resize(info_panel,
                                      (info_panel.shape[1] // scale,
                                       info_panel.shape[0] // scale))
        # put on the top right
        offset_x = undistorted.shape[1] - info_panel_small.shape[1]
        undistorted[0:info_panel_small.shape[0], offset_x:] = info_panel_small

        # Recast the x and y points into usable format for cv2.fillPoly()
        unwarp_zero = np.zeros_like(binary_unwarped).astype(np.uint8)
        color_unwarp = np.dstack((unwarp_zero, unwarp_zero, unwarp_zero))
        pts_left = np.array([
            np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([
            np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_unwarp, np.int_([pts]), (0, 255, 0))
        warped_road = cv2.warpPerspective(color_unwarp,
                                          self.unwarper.inv_M, img_size)

        # get the curvature and put text
        left_curve = left_line.radius_of_curvature
        right_curve = right_line.radius_of_curvature

        avg_curve = (left_curve + right_curve) / 2
        curvature_txt = "Radius of Curvature = {:.0f}(m)".format(avg_curve)
        deviation = Line.deviate_of_center(left_line, right_line)
        if deviation >= 0:
            dtxt = "Vechicle is {:.2f}(m) right of center".format(deviation)
        else:
            dtxt = "Vechicle is {:.2f}(m) left of center".format(-deviation)

        cv2.putText(undistorted, curvature_txt, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(undistorted, dtxt, (100, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, warped_road, 0.3, 0)

        return result


