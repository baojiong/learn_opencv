import numpy as np
import imutils
import cv2

image = cv2.imread("./test.png")
cv2.imshow('Original', image)

(h, w) = image.shape[:2]
center = (w//2, h//2)

cv2.waitKey(0)

M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2. imshow("Rotated by 45 Degrees", rotated)

cv2.waitKey(0)

M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2. imshow("Rotated by -90 Degrees", rotated)

cv2.waitKey(0)

rotated = imutils.rotate(image, 180)
cv2.imshow("Rotated by 180 Degrees", rotated)

cv2.waitKey(0)
