import shutil   # offers high-level operation on a file like a copy, create, and remote operation on the file
from os import listdir, path, rename
from tkinter import Frame
from unittest import result   #Get file name in Tkinter
import cv2

validateFolder = "../../dataset/validate_image"
trainFolder = "../../dataset/train_image"
segmentFolder = "../../dataset/segmentation_labels"

f = open("../docs/image_segment.txt", "w")

def map_link(fname_x, folder):
    for fname in fname_x:
        if fname.endswith(".jpg"):
            if(folder == 1):
                name = validateFolder
            else:
                name = trainFolder
            result = name + "/" + fname
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

if __name__ == '__main__':
    map_link(listdir(validateFolder), 1)
    map_link(listdir(trainFolder), 2)