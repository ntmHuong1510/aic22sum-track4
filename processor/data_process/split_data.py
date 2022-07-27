from itertools import count
import shutil   # offers high-level operation on a file like a copy, create, and remote operation on the file
from os import listdir, path, rename
import cv2

fnames_train = listdir('../labels')
source_train  = "../labels/"
target_train = "../dataset/train_labels/"


count = 1
num_img = (len(fnames_train) / 100 ) * 80
for fname in fnames_train:
    shutil.copyfile(source_train + fname, target_train + fname)
    count = count + 1
    if(count > num_img):
        break
count = 1
for fname in fnames_train:
    count = count + 1
    if(count > num_img):
        shutil.copyfile(source_train + fname, "../dataset/validate_labels/" + fname)

fnames_train = listdir('../images')
source_train  = "../images/"
target_train = "../dataset/train_images/"


count = 1
for fname in fnames_train:
    shutil.copyfile(source_train + fname, target_train + fname)
    count = count + 1
    if(count > num_img):
        break

count = 1
for fname in fnames_train:
    count = count + 1
    if(count > num_img):
        shutil.copyfile(source_train + fname, "../dataset/validate_images/" + fname)