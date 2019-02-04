import cv2
from imutils.video import VideoStream
from imutils.video import FPS

import numpy as np
import argparse
import imutils
import time
import cv2
import math

cap = cv2.VideoCapture("/home/vipul/Project/real-time-object-detection/HD Surveillance Video Sample - Click Settings 1080p to view HD quality.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("./home/vipul/Project/real-time-object-detection/HD Surveillance Video Sample - Click Settings 1080p to view HD quality.mp4")
    cv2.waitKey(1000)
    print ("Wait for the header")

post_frame = cap.get(1)
while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        cv2.imshow('video', frame)
        post_frame = cap.get(1)
        print (str(post_frame)+" frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(1, pos_frame-1)
        print ("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(1) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
