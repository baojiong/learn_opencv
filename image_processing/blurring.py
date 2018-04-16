import numpy as np
import cv2
import imutils

image = cv2.imread("../faces.jpeg")
#cv2.imshow('Original', image)

resized = imutils.resize(image, width=500)

blurred = np.hstack([
    cv2.blur(resized, (3, 3)),
    cv2.blur(resized, (5, 5)),
    cv2.blur(resized, (7, 7))])

cv2.imshow("Averaged", blurred)
cv2.waitKey(0)

blurred = np.hstack([cv2.GaussianBlur(resized, (3, 3), 0),
                     cv2.GaussianBlur(resized, (5, 5), 0),
                     cv2.GaussianBlur(resized, (7, 7), 0)])
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)

blurred = np.hstack([cv2.medianBlur(resized, 3),
                     cv2.medianBlur(resized, 5),
                     cv2.medianBlur(resized, 7)])
cv2.imshow("Median", blurred)
cv2.waitKey(0)

blurred = np.hstack([cv2.bilateralFilter(resized, 5, 21, 21),
                     cv2.bilateralFilter(resized, 7, 31, 31),
                     cv2.bilateralFilter(resized, 9, 41, 41)])
cv2.imshow("Bilateral", blurred)
cv2.waitKey(0)
