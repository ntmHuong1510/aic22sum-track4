import shutil   # offers high-level operation on a file like a copy, create, and remote operation on the file
from os import listdir, path, rename
from tkinter import Frame
from unittest import result   #Get file name in Tkinter
import cv2

fname1 = listdir("../background/testA_1")
f = open("../docs/link_background.txt", "w")

for fname in fname1:
    f.write("../background/testA_1/" + fname + '\n')