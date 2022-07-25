import shutil   # offers high-level operation on a file like a copy, create, and remote operation on the file
from os import listdir, path, rename
from tkinter import Frame
from unittest import result   #Get file name in Tkinter
import cv2

validateFolder = "../../dataset/validate_image"
trainFolder = "../../dataset/train_image"
segmentFolder = "../../dataset/segmentation_labels"

fname1 = listdir(validateFolder)
fname2 = listdir(trainFolder)
f = open("image_segment.txt", "w")
step = 1
count = 0

    ### Get the path link of image and corresponding segment
for fname in fname1:
    if fname.endswith(".jpg"):
        result = validateFolder + "/" + fname
        list = fname.split("_")
        first = list[0]
        tmp = list[1].split(".")
        second = tmp[0] #second idx in file name
        idx = int(first)    # index of 
        idxSub = int(second)
        filename = str(int(idx + 1)).zfill(5) + "_" + str(idxSub + 1) + "_seg.jpg"
        line = result + " " + segmentFolder + "/" +filename + "\n"
        print(first)
        f.write(str(idx) + ":" + line)

for fname in fname2:
    if fname.endswith(".jpg"):
        result = validateFolder + "/" + fname
        list = fname.split("_")
        first = list[0]
        tmp = list[1].split(".")
        second = tmp[0] #second idx in file name
        idx = int(first)    # index of 
        idxSub = int(second)
        filename = str(int(idx + 1)).zfill(5) + "_" + str(idxSub + 1) + "_seg.jpg"
        line = result + " " + segmentFolder + "/" +filename + "\n"
        print(first)
        f.write(str(idx) + ":" + line)
