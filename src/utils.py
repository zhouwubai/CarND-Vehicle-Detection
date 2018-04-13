import cv2
import numpy as np


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
    alive_box = []
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
                alive_box.append(bbox)
        else:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
            alive_box.append(bbox)
    # Return the image
    return img, alive_box


def eval_poly(coeffs, y):
    return (coeffs[0] * y ** 2 + coeffs[1] * y + coeffs[2])


def curvature_poly(coeffs, y):
    """
    calculate the curvature of a polyline at point y
    """
    a, b = coeffs[0], coeffs[1]
    return ((1 + (2 * a * y + b) ** 2) ** 1.5 / np.absolute(2 * a))


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, inplace=False):
    if inplace:
        draw_img = img
    else:
        draw_img = np.copy(img)
    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img



