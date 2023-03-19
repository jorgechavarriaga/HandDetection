import cv2
from math import dist
import mediapipe as mp
import utils.BGRColor as BGR

class handDetector():
    def __init__(self, staticImage = False, maxHands = 2, modelComplex = 1, 
                 minDetectionConfidence = 0.75, minTrackingConfidence = 0.75):
        self.staticImage            = staticImage
        self.maxHands               = maxHands
        self.modelComplex           = modelComplex
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence  = minTrackingConfidence
        self.mpHands                = mp.solutions.hands
        self.hands                  = self.mpHands.Hands(self.staticImage, self.maxHands, 
                                        self.modelComplex, self.minDetectionConfidence, 
                                        self.minTrackingConfidence)
        self.mpDrawing              = mp.solutions.drawing_utils
        self.mpDrawingStyles        = mp.solutions.drawing_styles

    def findHands(self, img, drawImage = True):
        imgToRGB                = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results            = self.hands.process(imgToRGB)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if drawImage:
                    self.mpDrawing.draw_landmarks(img, handLandmarks,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDrawing.DrawingSpec(color = BGR.YELLOW, thickness = 2, circle_radius = 2),
                        self.mpDrawing.DrawingSpec(color = BGR.GREEN, thickness = 2, circle_radius = 2))
        return img
    
    def findPosition(self, img, handNumber = 0, drawImage = True):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            hands = self.results.multi_hand_landmarks[handNumber]
            for keypoint, handLandmark in enumerate(hands.landmark):
                height, width, _ = img.shape
                cx , cy = int(handLandmark.x * width), int(handLandmark.y * height)
                landmarkList.append([keypoint, cx, cy])
                if drawImage:
                    cv2.circle(img, (cx, cy), 8, BGR.YELLOW, cv2.FILLED )
        return landmarkList
    
    def distance(self, landmarkList, finger1, finger2):
        cxFinger1, cyFinger1    = 0 , 0
        cxFinger2, cyFinger2    = 0 , 0
        if len(landmarkList) !=0:
            for i in landmarkList:
                if i[0] == finger1:
                    cxFinger1, cyFinger1 = i[1], i[2]
                if i[0] == finger2: 
                    cxFinger2, cyFinger2 = i[1], i[2]
        distance = dist((cxFinger1,cyFinger1),(cxFinger2,cyFinger2))
        return distance
