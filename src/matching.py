import numpy as np
import cv2
import matplotlib.image as mpimg


def template_match(img, template_list, method=cv2.TM_CCOEFF):
    bbox_list = []
    for template in template_list:
        template_img = mpimg.imread(template)
        h, w, _ = template_img.shape
        res = cv2.matchTemplate(img, template_img, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bbox_list.append((top_left, bottom_right))
    return bbox_list


def template_multi_match(img, template_list,
                         threshold, method=cv2.TM_CCOEFF):
    bbox_list = []
    for template in template_list:
        template_img = mpimg.imread(template)
        h, w, _ = template_img.shape
        res = cv2.matchTemplate(img, template_img, method)

        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            bottom_right = (pt[0] + w, pt[1] + h)
            bbox_list.append((pt, bottom_right))
    return bbox_list


