# AI-Volume-Gesture-Control
just by using Hand Gesture we can alter the volume. using python and opencv amd the handtracking module. This repository contains my VolumeHandControl.py Code my modules for hand tracking

## Volume Control by Hand Gestures
      import cv2
      import time
      import numpy as np
      import HandTrackingModule as htm
      import math
      from ctypes import cast, POINTER
      from comtypes import CLSCTX_ALL
      from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

      #####################################
      wCam, hCam = 640, 480
      #####################################
      cap = cv2.VideoCapture(0)
      cap.set(3, wCam)
      cap.set(4, hCam)
      pTime = 0
      volPer = 0
      detector = htm.handDetector(detectionCon=0.8)

      devices = AudioUtilities.GetSpeakers()
      interface = devices.Activate(
          IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
      volume = cast(interface, POINTER(IAudioEndpointVolume))
      # volume.GetMute()
      # volume.GetMasterVolumeLevel()
      volRange = volume.GetVolumeRange()

      minVol = volRange[0]
      maxVol = volRange[1]
      vol = 0
      volBar = 400
      while True:
          success, img = cap.read()
          img = detector.findHands(img)
          lmList = detector.findPostion(img, draw=False)
          if len(lmList) != 0:
              # print(lmList[4], lmList[8])
              x1, y1 = lmList[4][1], lmList[4][2]
              x2, y2 = lmList[8][1], lmList[8][2]
              cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

              cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
              cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
              cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
              cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

              length = math.hypot(x2 - x1, y2 - y1)
              # print(length)
              # hand range min(Track points) 50 - max(track points) 350
              # Volume Range  -74  - 0
              vol = np.interp(length, [50, 220], [minVol, maxVol])
              volBar = np.interp(length, [50, 220], [400, 150])
              volPer = np.interp(length, [50, 220], [0, 100])
              print(int(length), vol)
              volume.SetMasterVolumeLevel(vol, None)

              if length < 50:
                  cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

          cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
          cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
          cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

          cTime = time.time()
          fps = 1 / (cTime - pTime)
          pTime = cTime
          cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

          cv2.imshow("img", img)
          cv2.waitKey(1)


## HandTracking Module (to track movement of hands)

    import mediapipe as mp
    import time

    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
           for handLms in results.multi_hand_landmarks:
               for id,lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    print(id, cx , cy)
                    if id == 0:
                        cv2.circle(img, (cx,cy), 20, (255,0,255), cv2.FILLED)
               mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        cv2.imshow("image", img)
        cv2.waitKey(1)
       
## MY Hand Tracking Module which can be used for any project related to gestures

      import cv2
      import mediapipe as mp
      import time

      class handDetector():
          def __init__(self,mode=False, maxHands = 2, detectionCon=0.5, trackCon=0.5 ):
             self.mode = mode
             self.maxHands = maxHands
             self.detectionCon = detectionCon
             self.trackCon = trackCon

             self.mpHands = mp.solutions.hands
             self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
             self.mpDraw = mp.solutions.drawing_utils

          def findHands(self, img,draw=True):
              imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              self.results = self.hands.process(imgRGB)#
              # print(results.multi_hand_landmarks)

              if self.results.multi_hand_landmarks:
                  for handLms in self.results.multi_hand_landmarks:
                      if draw:
                          self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
              return img

          def findPostion(self, img, handNo=0, draw = True):
              lmList = []
              if self.results.multi_hand_landmarks:
                  myHand = self.results.multi_hand_landmarks[handNo]
                  for id, lm in enumerate(myHand.landmark):
                      # print(id,lm)
                      h, w, c = img.shape
                      cx, cy = int(lm.x * w), int(lm.y * h)
                      # print(id, cx, cy)
                      lmList.append([id, cx, cy])
                      if draw:
                          cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

              return lmList


      def main():
          pTime = 0
          cTime = 0
          cap = cv2.VideoCapture(0)
          detector = handDetector()
          while True:
              success, img = cap.read()
              img = detector.findHands(img)
              lmList = detector.findPostion(img)
              if len(lmList) != 0:
                  print(lmList[4])
              cTime = time.time()
              fps = 1 / (cTime - pTime)
              pTime = cTime
              cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
              cv2.imshow("image", img)
              cv2.waitKey(1)


      if __name__ == "__main__":
          main()

