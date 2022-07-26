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


fsave = listdir("../data_generate_sample/")
def get_number_image_per_class(file):
    arr = []
    for filename in file:
        arr.append(int(filename.split(" ")[1].split("\n")[0])) # get the number of image in each class
    return arr

def get_link_img_seg(file="../link_img_seg_per_class/"):
    arr = []
    for i in range(115):
        arr.append(file + str(i).zfill(5) + ".txt") 
    return arr

def resize_img(img, bg, mask):
    """
        if size img >= size bg ==> Scale size of img with its ratio width/height
    """
    while img.shape[0] >= bg.shape[0] or img.shape[1] >= bg.shape[1]: 
        height_new = np.random.randint(1, img.shape[0] * 0.7)   #random size height of image
        height_percent = (height_new / float(img.shape[0])) #get its percentage according to old height and new height
        width_new = int((float(img.shape[1]) * float(height_percent)))  # compute new width base on its ratio 
        img = cv2.resize(img, (width_new, height_new), interpolation = cv2.INTER_AREA)  #==> Resize img and its mask
        width_mask = int((float(mask.shape[1]) * float(height_percent)))
        mask = cv2.resize(mask, (width_mask, height_new), interpolation = cv2.INTER_AREA)
    return img.shape[:2]

def get_img_and_mask(link_file):
    # format link_file:  link_img link_mask\n
    img  = cv2.imread(link_file.split(" ")[0])
    mask = cv2.imread(link_file.split(" ")[1].split("\n")[0])
    mask[mask < 255] = 0    #Adjust mask with 2 value: 0 and 255
    mask = mask.astype(np.uint8)
    return img, mask

def random_coordinate(img, bg, overlap_percent):
    """
        if percentage that 2 objects ovelap, we must random other coordinate for objs
    """
    img0_cor, img1_cor = img.shape[:2]
    while(overlap_percent >= 0.5):
        x, y = np.random.randint(0, bg.shape[1] - img1_cor - 1), np.random.randint(0, bg.shape[0] - img0_cor - 1)   #generate a coordianate in background image
        crop_img = bg[y: y + img0_cor, x: x + img1_cor]
        sum_bg = np.sum(crop_img) - np.sum(crop_img[crop_img < 255])
        if(sum_bg == 0):
            break
        min_value = np.minimum(bg[y: y + img0_cor, x: x + img1_cor], mask)  #Get overlapping region
        min_value = np.sum(min_value) - np.sum(min_value[min_value < 255])
        overlap_percent = min_value / sum_bg
    return x, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='linkpath_image_segment.txt')
    args = parser.parse_args()
    file = copen(args.filename)

    arr = get_link_img_seg()
    
    cv_limit = 512
    num_img = 20
    count = 0
    number_background = 1
    for i in range(number_background):
        bg2 = cv2.imread('12.jpg')
        bg = np.zeros((bg2.shape[0], bg2.shape[1], 3), dtype = "uint8")
        while(True):
            filename = open(arr[(count - 1) % len(arr)], "r")   #get file path of each class in order
            lines = filename.readlines()
            count = count + 1
            img, mask = get_img_and_mask(lines[0])  # read the first image in class
            img_new_size = resize_img(img, bg, mask)    #check boundary of img and scale it into background region

            img0_cor, img1_cor = img.shape[:2]
            x, y = random_coordinate(img, bg, 1)
            bg[y: y + img0_cor, x: x + img1_cor][mask == 255] = 255     #assign mask into black bg
            bg2[y: y + img0_cor, x: x + img1_cor][mask ==255] = img[mask == 255]    #assign object in img into background
            if(count == num_img):
                count = 0
                break
        cv2.imshow("origin", bg)
        cv2.waitKey()
        cv2.imshow("background ", bg2)
        cv2.waitKey()
        # cv2.imwrite("../data_generate_sample/bg_" + str(i).zfill(5) + ".jpg", bg)
