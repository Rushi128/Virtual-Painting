import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# this is for webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)
# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)
# find Functions
# x is the raw distance y is thevalue in the cm that is measured by the teap
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)


#loops
while True:
    Success, img = cap.read()
    hands = detector.findHands(img, draw=False)
    if hands:
        lmList = hands[0]['lmList']
        x ,y, w, h = hands[0]['bbox']
        x1, y1 = lmList[5]
        x2, y2 = lmList[17]

        distance = int(math.sqrt ((y2-y1) ** 2 + (x2-x1) ** 2))
        A, B, C = coff
        distanceCM = int(A*distance**2 + B*distance + C)

        # print(distanceCM , distance)
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x, y))

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3)
        
    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break