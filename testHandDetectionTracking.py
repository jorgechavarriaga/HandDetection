import os
import cv2 
import time 
import mediapipe as mp
import utils.HandDetectionTracking as hdt
import utils.Finger as Finger
import utils.BGRColor as BGR

os.system('cls')

pTime, cTime            = 0, 0
cap                     = cv2.VideoCapture(0)
detector                = hdt.handDetector()
finger1                 = Finger.THUMB
finger2                 = Finger.PINKY

while cap.isOpened():
    _, img              = cap.read()
    img                 = detector.findHands(img)
    landmarkList        = detector.findPosition(img)
    distance            = detector.distance(landmarkList, finger1, finger2)
    cTime               = time.time()
    fps                 = 1 / (cTime - pTime)
    pTime               = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, BGR.RED, 1, cv2.LINE_AA )
    cv2.putText(img, f'DISTANCE: {int(distance)}', (20,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, BGR.RED, 1, cv2.LINE_AA )
    cv2.imshow('Hands Detector & Tracking', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()