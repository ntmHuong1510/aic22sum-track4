from fileinput import filename
from os import listdir, path, rename
import random
from random import randint
from linereader import copen
import numpy as np
import cv2
import argparse
import math as m

import torch

#Lay sum cua cai hinh nho chia cho hinh lon

fsave = listdir("../data_generate_sample/")
def get_number_image_per_class(file):
    arr = []
    for filename in file:
        arr.append(int(filename.split(" ")[1].split("\n")[0])) # get the number of image in each class
    return arr

def get_file_link_img_seg_per_class(file="../link_img_seg_per_class/"):
    arr = []
    for i in range(115):
        arr.append(file + str(i).zfill(5) + ".txt") 
    return arr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='linkpath_image_segment.txt')
    args = parser.parse_args()  #"image_per_class.txt"
    file = copen(args.filename)

    # arr = get_number_image_per_class(file)
    arr = get_file_link_img_seg_per_class()
    
    cv_limit = 512
    num_img = 20
    count = 0
    number_background = 1
    for i in range(number_background):
        # bg = cv2.imread('12.jpg')
        bg2 = cv2.imread('12.jpg')
        bg = np.zeros((bg2.shape[0], bg2.shape[1], 3), dtype = "uint8")
        # cv2.imshow('Single Channel Window', bg)
        # cv2.waitKey(0)
        while(True):
            filename = open(arr[(count - 1) % len(arr)], "r")   #get file path of each class in order
            lines = filename.readlines()
            link = lines[0]  # read the first image in class
           
            count = count + 1
            img  = cv2.imread(link.split(" ")[0])
            mask = cv2.imread(link.split(" ")[1].split("\n")[0])
            mask[mask < 255] = 0
            mask = mask.astype(np.uint8)
            img0_cor, img1_cor = img.shape[:2]
            ## if size of the image is greate than size of background, we must scale it inter background area by the way 
            ## get height in range of [1, 2/3(height_original obj)] and then estimate width according to this new percentage
            ## to make sure the ratio width/height of image is not change
            while img0_cor >= bg.shape[0] or img1_cor >= bg.shape[1]: 
                height_new = np.random.randint(1, img.shape[0] * 0.7)
                height_percent = (height_new / float(img.shape[0]))
                width_new = int((float(img.shape[1]) * float(height_percent)))
                img = cv2.resize(img, (width_new, height_new), interpolation = cv2.INTER_AREA)
                width_mask = int((float(mask.shape[1]) * float(height_percent)))
                mask = cv2.resize(mask, (width_mask, height_new), interpolation = cv2.INTER_AREA)

            img0_cor, img1_cor = img.shape[:2]
            percen_overlap = 1
            while(percen_overlap >= 0.5):
                x, y = np.random.randint(0, bg.shape[1] - img1_cor - 1), np.random.randint(0, bg.shape[0] - img0_cor - 1)   #generate a coordianate in background image
                crop_img = bg[y: y + img0_cor, x: x + img1_cor]
                sum_bg = np.sum(crop_img)  - np.sum(crop_img[crop_img < 255])
                if(sum_bg == 0):
                    break
                min_value = np.minimum(bg[y: y + img0_cor, x: x + img1_cor], mask)
                min_value = np.sum(min_value) - np.sum(min_value[min_value < 255])
                percen_overlap = min_value /  sum_bg
                print(percen_overlap)
            bg[y: y + img0_cor, x: x + img1_cor][mask == 255] = 255
            bg2[y: y + img0_cor, x: x + img1_cor][mask ==255] = img[mask == 255]
            if(count == num_img):
                count = 0
                break
        cv2.imshow("origin", bg)
        cv2.waitKey()
        cv2.imshow("background ", bg2)
        cv2.waitKey()
        # cv2.imwrite("../data_generate_sample/bg_" + str(i).zfill(5) + ".jpg", bg)
