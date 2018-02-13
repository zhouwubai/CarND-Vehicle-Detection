import numpy as np
from line.base import Line


def search_lines(binary_warped, nwindows=9, margin=100,
                 minpix=50, draw_windows=False):
    """
    Given a unwarped image, return the fitted two lines
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # choose the number of sliding windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    left_windows, right_windows = [], []

    # step through the windows one by one
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if draw_windows:
            left_windows.append(((win_xleft_low, win_y_low),
                                (win_xleft_high, win_y_high)))
            right_windows.append(((win_xright_low, win_y_low),
                                 (win_xright_high, win_y_high)))

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # if you found > minpix pixels
        # recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # return left line and right line
    left_line, right_line = Line(), Line()
    left_line.fit(leftx, lefty)
    right_line.fit(rightx, righty)
    if draw_windows:
        return (left_line, right_line, left_windows, right_windows)
    else:
        return left_line, right_line


def fast_search_lines(binary_warped, left_fit, right_fit,
                      margin=100, minpix=100):
    """
    here minpix is different from previous minpix
    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) +
                                   left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                   left_fit[1] * nonzeroy +
                                   left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) +
                                    right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                    right_fit[1] * nonzeroy +
                                    right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # return left line and right line
    left_line, right_line = Line(), Line()
    left_line.fit(leftx, lefty)
    right_line.fit(rightx, righty)
    return left_line, right_line


