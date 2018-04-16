from __future__ import print_function
import argparse
import cv2

#ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required = True, help = "Path to the image")
#args = vars(ap.parse_args())

image = cv2.imread("../faces.jpeg")

(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))
cv2.waitKey(0)

corner = image[0:100, 0:100]
cv2.imshow("Corner", corner)
cv2.waitKey(0)

image[0:100, 0:100] = (0, 255, 0)
cv2.imshow("Update", image)
cv2.waitKey(0)

