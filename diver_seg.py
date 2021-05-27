import numpy as np
import cv2
import pandas as pd
from tabulate import tabulate

cap = cv2.VideoCapture('IMG_1804__2175.m4v')

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
#
# size = (frame_width, frame_height)
#
# result = cv2.VideoWriter('filename.avi',                  # to write the video to a file
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))     # structuring element
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# fgbg = cv2.createBackgroundSubtractorMOG2(history=350, varThreshold=45, detectShadows=False)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()           # background subtractor
# history=350, varThreshold=45, detectShadows=False
# 500, 16, True/False

total_frames = 0
brightest_image = np.ones(shape=(1280, 720))
outp = []
while 1:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)      # applies background subtractor to every frame
    # blur = cv2.GaussianBlur(fgmask, (5, 5), 0)
    # ret1, thresh_img = cv2.threshold(blur, 91, 255, cv2.THRESH_BINARY)
    if total_frames != 0:
        pass
    else:
        total_frames += 1
        # blend = cv2.addWeighted(brightest_image, 0.5, frame, 0.8, 0.0)
        continue
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_ERODE, kernel)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel1)

    # fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
    # xy = np.sum(fgmask)
    xy = cv2.sumElems(fgmask)               # calculates brightness of frame
    if xy > cv2.sumElems(brightest_image):
        brightest_image = fgmask

    edged = cv2.Canny(frame, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # finds contours

    if len(contours) != 0:
        cv2.drawContours(fgmask, contours, -1, (0, 255, 0), 3)  # draws contours

        c = max(contours, key=cv2.contourArea)

        # for c1 in c:
        #     cv2.drawContours(fgmask, [c1], -1, (0, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(fgmask, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if total_frames % 6 == 0:
        # outp.append(brightest_image)
        # addition = np.add(outp[len(outp) - 1], outp[len(outp) - 2])
        # pos = np.where(addition == 2)
        # print(addition[pos])
        # if len(pos) == 0:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 600, 600)             # resizes the window
        cv2.imshow('frame', fgmask)
            # result.write(fgmask)
        # pos = 0
        # brightest_image = np.zeros(shape=(1280, 720))

    total_frames += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print(total_frames)
        break

# result.release()
cap.release()
cv2.destroyAllWindows()
