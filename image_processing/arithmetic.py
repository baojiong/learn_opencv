import numpy as np
import cv2
import imutils

image = cv2.imread("./test.png")
cv2.imshow('Original', image)
cv2.waitKey(0)

print ("max of 255: {}".format(cv2.add(np.uint8([200]), np.uint8([100]))))
print ("min of 0: {}".format(cv2.subtract(np.uint8([50]), np.uint8([100]))))
print ("wrap around: {}".format(np.uint8([200]) + np.uint8([100])))
print ("wrap around: {}".format(np.uint8([50]) - np.uint8([100])))

m = np.ones(image.shape, dtype="uint8") * 100
added = cv2.add(image, m)
cv2.imshow("Added by CV2", added)
cv2.waitKey(0)

m = np.ones(image.shape, dtype="uint8") * 100
added = image + m
cv2.imshow("Added by Numpy", added)
cv2.waitKey(0)

m = np.ones(image.shape, dtype="uint8") * 50
subtracted = cv2.subtract(image, m)
cv2.imshow("Subtracted by CV2", subtracted)
cv2.waitKey(0)

m = np.ones(image.shape, dtype="uint8") * 50
subtracted = image - m
cv2.imshow("Subtracted by Numpy", subtracted)
cv2.waitKey(0)

