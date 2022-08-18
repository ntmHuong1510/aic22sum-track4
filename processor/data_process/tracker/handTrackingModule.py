import imp
from operator import mod
import cv2
import mediapipe as mp

class handTracker():
    """
        Create class use for tracking with basic parameters required for the hands funtions to work
    """
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplexity = modelComplexity
        self.trackCon = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    """
        Creating a method that will track the hands in our input image
                and draw the hand connection
    """
    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLm, self.mpHands.HAND_CONNECTIONS)
        return image       
    """
        Creating a method to find the ‘x’ and ‘y’ coordinates of each hand point
    """
    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)   #each point in the hand lanmark diagram
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
        return lmlist

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success,image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            print(lmList[4])

        cv2.imshow("Video",image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()