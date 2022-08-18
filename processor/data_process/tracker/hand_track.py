from xml.sax import make_parser
import cv2
import mediapipe as mp

#### Importations and initializations

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#### Capturing an image input and processing it
"""
    Process the RGB image to identify the hands in the image
"""
while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    ### Working with each hand
    #Checking whether a hand is detected
    if results.multi_hand_landmarks:    #Check whether a hand is detected
        for handLms in results.multi_hand_landmarks:    # Working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)   #each point in the hand lanmark diagram
                #### Drawing the hand landmarks and hand connections on the hand image
                if id == 20:
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
        #### Displaying the output
        cv2.imshow("Output", image)
        cv2.waitKey(1)