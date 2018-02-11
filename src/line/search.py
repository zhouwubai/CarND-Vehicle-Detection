import numpy as np
import cv2


def find_lines(binary_warped, nwindows=9, margin=100,
               minpix=50, draw_windows=False):
    """
    Given a unwarped image, return the fitted two lines
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = binary_warped

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
            cv2.rectangle(out_img,
                          (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high),
                          (255, 0, 0), 5)
            cv2.rectangle(out_img,
                          (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high),
                          (0, 0, 255), 5)

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

    leftx, lefty, left_fit = None, None, None
    if len(left_lane_inds) >= minpix:  # at least one window search success
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)

    rightx, righty, right_fit = None, None, None
    if len(right_lane_inds) >= minpix:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fit = np.polyfit(righty, rightx, 2)

    # Fit a second order polynomial to each
    return left_fit, right_fit, leftx, lefty, rightx, righty


def find_lines_with_previous(binary_warped, left_fit, right_fit,
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

    # Again, extract left and right line pixel positions
    leftx, lefty, left_fit = None, None, None
    if len(left_lane_inds) >= minpix:  # at least one window search success
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        if lefty.size == 0 or leftx.size == 0:
            print(left_fit, right_fit, lefty, leftx)
        left_fit = np.polyfit(lefty, leftx, 2)

    rightx, righty, right_fit = None, None, None
    if len(right_lane_inds) >= minpix:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fit = np.polyfit(righty, rightx, 2)

    # Fit a second order polynomial to each
    return left_fit, right_fit, leftx, lefty, rightx, righty


def curvature(a, b, c, y):
    return ((1 + (2 * a * y + b) ** 2) ** 1.5 / np.absolute(2 * a))


def curvature_in_pixel(fit, y):
    """
    fit: numpy array of three a, b, c
    """
    return curvature(fit[0], fit[1], fit[2], y)


def curvature_pixel_2_meter(fit, y,
                            ym_per_pix=30.0 / 720,
                            xm_per_pix=3.7 / 700):
    """
    fit:
        the parabola in pixel
    y:
        y in pixel
    """
    return curvature(fit[0] * xm_per_pix / (ym_per_pix ** 2),
                     fit[1] * xm_per_pix / ym_per_pix, fit[2],
                     y * ym_per_pix)


def evaluate(fitx, y):
    return (fitx[0] * y ** 2 + fitx[1] * y + fitx[2])


def deviate_of_center(left_fit, right_fit, shape):
    """
    positve: right
    negative: left
    """
    y, x = shape[0], shape[1]
    x_left, x_right = evaluate(left_fit, y), evaluate(right_fit, y)
    return (x / 2.0 - (x_left + x_right) / 2.0)



