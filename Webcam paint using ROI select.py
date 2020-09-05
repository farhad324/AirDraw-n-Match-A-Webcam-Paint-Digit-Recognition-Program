# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:54:05 2020

@author: farhad324
"""


import cv2
import numpy as np

tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture(0)

while True:
    k,frame = video.read()
    frame = cv2.flip(frame,1)
    cv2.imshow("Tracking",frame)
    if cv2.waitKey(10) == ord('q'):
        break
bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)
cv2.destroyWindow("ROI selector")
img=np.zeros((512,512,3),np.int8)
while True:
    ok, frame = video.read()
    frame = cv2.flip(frame,1)
    ok, bbox = tracker.update(frame)

    if ok:
        x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cv2.rectangle (frame,(x,y),((x+w),(y+h)),(255,0,0),3,1)
        cv2.putText(frame, "Tracking", (75,75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,10,0), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Lost", (75,75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,10,0), 1, cv2.LINE_AA)
    cv2.imshow("Tracking", frame)
    cv2.imshow('Canvas',img)
    cv2.circle(img,(x,y),6,(256,256,256),-1)
    cv2.circle(frame,(x,y),6,(256,256,256),2)
    if cv2.waitKey(10) == ord('q'):
        break
