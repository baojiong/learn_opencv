import numpy as np
import cv2

image = cv2.imread("../coins.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = image - np.ones(image.shape, dtype="uint8") * 10
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

(T, threshInv) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Invent", threshInv)

cv2.imshow("Coins", cv2.bitwise_and(image, image, mask=threshInv))

cv2.waitKey(0)