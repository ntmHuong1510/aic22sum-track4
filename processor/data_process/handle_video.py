import cv2
import os

rel_path = "..\\..\\Dataset\\AIC22_Track4_TestA\\TestA\\" 

videos = [rel_path + "testA_1.mp4", rel_path + "testA_2.mp4",rel_path +  "testA_3.mp4",rel_path +  "testA_4.mp4",rel_path + "testA_5.mp4"]
idx = 0
cap = cv2.VideoCapture(videos[idx])
ret, frame = cap.read()
while(1):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key == ord('q') or ret==False:
        cap.release()
        cv2.destroyAllWindows()
        break
    elif key == ord('p'):
        cv2.waitKey(0)
    elif key == ord('f'):
        print(idx, idx % len(videos))
        idx = idx + 1
        cap = cv2.VideoCapture(videos[idx % len(videos)])
    elif key == ord('b'):
        if idx != 0:
            idx  = idx - 1
        else: 
            break
        cap = cv2.VideoCapture(videos[idx % len(videos)])
    cv2.imshow('frame',frame)