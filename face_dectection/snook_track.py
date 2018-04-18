import numpy as np
import argparse
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

blueLower = np.array([100, 67, 0], dtype="uint8")
blueUpper = np.array([255, 128, 50], dtype="uint8")

redLower = np.array([0, 67, 100], dtype="uint8")
redUpper = np.array([50, 128, 255], dtype="uint8")

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
        break

    blue = cv2.inRange(frame, blueLower, blueUpper)
    blue = cv2.GaussianBlur(blue, (3, 3), 0)

    red = cv2.inRange(frame, redLower, redUpper)
    red = cv2.GaussianBlur(red, (3, 3), 0)

    (_, cnts_blue, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, cnts_red, _) = cv2.findContours(red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts_blue) > 0:
        cnt_blue = sorted(cnts_blue, key=cv2.contourArea, reverse=True)[0]

        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt_blue)))
        cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

    if len(cnts_red) > 0:
        cnt_red = sorted(cnts_red, key=cv2.contourArea, reverse=True)[0]

        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt_red)))
        cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Binary", blue + red)

    time.sleep(0.025)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()