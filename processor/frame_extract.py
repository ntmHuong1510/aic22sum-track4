# Importing all necessary libraries 
import cv2 
import os 

# Read the video from specified path 
input_video = "testA_5"
cam = cv2.VideoCapture("../dataset/aic22_track4_testA_video/TestA/testA_5.mp4")   ##input video name

try: 
    # creating a folder named data 
    if not os.path.exists('./frame_extract/testA_5/'): 
        os.makedirs('./frame_extract/testA_5/') 

# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 

# frame 
currentframe = 0

while(True): 
    
    # reading from frame 
    ret,frame = cam.read() 

    if ret: 
        # if video is still left continue creating images 
        name = "./frame_extract/testA_5/" + str(currentframe).zfill(5) + '.jpg' 
        print ('Creating...' + name) 

        # writing the extracted images 
        cv2.imwrite(name, frame) 

        #counter to show which frame that are being process
        currentframe += 1
    else: 
        break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 