import shutil   # offers high-level operation on a file like a copy, create, and remote operation on the file
from os import listdir, path, rename
import cv2

fnames = listdir('../../dataset/segmentation_labels')
f = open("image_per_class.txt", "w")
step = 1
count = 0
for fname in fnames:
    if fname.endswith(".jpg"):
        list = fname.split("_")
        first = list[0]
        print(first)
        tmp = list[1].split(".")
        second = tmp[0] #second idx in file name
        idx = int(first)    # index of class
        if(idx != step):
            print(str(step) + "done!")
            f.write("{}: {}\n".format(step - 1, count))
            step = step + 1
            count = 0
        else:
            count = count + 1

