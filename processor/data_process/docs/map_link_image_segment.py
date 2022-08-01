import shutil   # offers high-level operation on a file like a copy, create, and remote operation on the file
from os import listdir, path, rename
from tkinter import Frame
from unittest import result   #Get file name in Tkinter
import cv2

validateFolder = "../../dataset/validate_image"
trainFolder = "../../dataset/train_image"
segmentFolder = "../../dataset/segmentation_labels"

f = open("../../docs/image_segment.txt", "w")

def map_link(fname_x, folder):
    for fname in fname_x:
        if fname.endswith(".jpg"):
            name = validateFolder if folder == 1 else trainFolder
            first = fname.split("_")[0]
            tmp = fname.split("_")[1].split(".")
            idx = int(first)    # index of 
            filename = path.join(str(int(idx + 1)).zfill(5), "_", str(int(tmp[0]) + 1), "_seg.jpg")
            line = path.join(name, "/", fname) + " " + segmentFolder + "/" + filename + "\n"
            print(first)
            f.write(str(idx) + ":" + line)

if __name__ == '__main__':
    map_link(listdir(validateFolder), 1)
    map_link(listdir(trainFolder), 2)