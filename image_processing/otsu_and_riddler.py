import numpy as np
import cv2
import mahotas

image = cv2.imread("../coins.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

T = mahotas.thresholding.otsu(blurred)
print("Otsu's threshold: {}".format(T))
