import cv2
import numpy as np
from image_processing import imutils


def detecteMan(filename):
    image = cv2.imread(filename, 0)
    image = imutils.resize(image, width=500)
    template = cv2.imread("temp.png", 0)
    r = image.shape[1] // 500
    template = imutils.resize(template, width=template.shape[1] * r)
    res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    cv2.rectangle(image, (min_loc[0], min_loc[1]), (min_loc[0] + template.shape[1], min_loc[1] + template.shape[0]), (0, 255, 0))
    cv2.imshow("bb", image)

    cv2.waitKey(0)


def detectCube(filename):
    image = cv2.imread(filename, 0)
    image = imutils.resize(image, width=500)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(image, 1, 10)
    cv2.imshow("bb", canny)

    cv2.waitKey(0)

"""

def get_center(img_canny, ):

    y_top = np.nonzero([max(row) for row in img_canny[400:]])[0][0] + 400
    x_top = int(np.mean(np.nonzero(canny_img[y_top])))

    y_bottom = y_top + 50
    for row in range(y_bottom, H):
        if canny_img[row, x_top] != 0:
            y_bottom = row
            break

    x_center, y_center = x_top, (y_top + y_bottom) // 2
    return img_canny, x_center, y_center
"""


def deskew(image, width):
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)

    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([
        [1, skew, -0.5 * w * skew],
        [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    image = imutils.resize(image, width=width)

    return image


image = cv2.imread("0.png", 0)
image = deskew(image, 500)
cv2.imshow("bb", image)
cv2.waitKey(0)



