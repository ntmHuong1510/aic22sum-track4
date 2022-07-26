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
from yaml import parse


def get_img_and_mask(link_file):
    # format link_file:  link_img link_mask\n
    img  = cv2.imread(link_file.split(" ")[0])
    mask = cv2.imread(link_file.split(" ")[1].split("\n")[0])
    mask[np.bitwise_and(mask < 255, mask > 200)] = 255    #Adjust mask with 2 value: 0 and 255
    mask = mask.astype(np.uint8)
    return img, mask

def get_number_image_per_class(file):
    arr = []
    for filename in file:
        arr.append(int(filename.split(" ")[1].split("\n")[0])) # get the number of image in each class
    return arr

def get_background(link_file="../docs/link_background.txt"):
    arr = []
    f = open(link_file)
    for filename in f:
        arr.append(filename.split("\n")[0]) # get the number of image in each class
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
    x, y = img.shape[:2]
    while x >= bg.shape[0] or y >= bg.shape[1]: 
        x = np.random.randint(1, x * 0.7)   #random size height of image
        height_percent = (x / float(img.shape[0])) #get its percentage according to old height and new height
        y = int((float(img.shape[1]) * float(height_percent)))  # compute new width base on its ratio 
        img = cv2.resize(img, (y, x), interpolation = cv2.INTER_AREA)  #==> Resize img and its mask
        width_mask = int((float(mask.shape[1]) * float(height_percent)))
        mask = cv2.resize(mask, (width_mask, x), interpolation = cv2.INTER_AREA)
        if img.shape[0] <= 1 or img.shape[1] <= 1:
            break

def random_coordinate(img, mask, bg, overlap_percent, ovl_thresh):
    """
        if percentage that 2 objects ovelap, we must random other coordinate for objs
    """
    img0_cor, img1_cor = img.shape[:2]
    x, y = -1, -1
    iter = 0
    for i  in range(7):
        overlap_percent = 1
        while(overlap_percent >= ovl_thresh and iter < 100):
            x, y = np.random.randint(0, bg.shape[1] - img1_cor - 1), np.random.randint(0, bg.shape[0] - img0_cor - 1)   #generate a coordianate in background image
            crop_img = bg[y: y + img0_cor, x: x + img1_cor]
            sum_bg = np.sum(crop_img) - np.sum(crop_img[crop_img < 255])
            if(sum_bg == 0):
                break
            min_value = np.minimum(bg[y: y + img0_cor, x: x + img1_cor], mask)  #Get overlapping region
            min_value = np.sum(min_value) - np.sum(min_value[min_value < 255])
            overlap_percent = min_value / sum_bg
            if(overlap_percent >= ovl_thresh):
                x = -1
            iter = iter + 1
        if(x != -1):
            break
        else:
            resize_img(img, bg, mask)
            if(img.shape[0] <= 1 or img.shape[1] <= 1):
                x = -1
                break

    return x, y

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='../docs/image_segment.txt')
    parser.add_argument('--bg_number', type=int, default=5800, help='number background need to generate data')
    parser.add_argument('--img_per_bg', type=int, default=20, help='number of objects generated in a background')
    parser.add_argument('--ovl_threshold', type=np.double, default=0.5, help='overlap threshold of 2 objects in an background')
    opt = parser.parse_args()
    return opt

def main(opt):
    file = copen(opt.filename)
    arr = get_link_img_seg()
    background = get_background()
    count = 1
    for i in range(opt.bg_number):
        bg2 = cv2.imread(background[(count - 1) % len(background)])
        bg = np.zeros((bg2.shape[0], bg2.shape[1], 3), dtype = "uint8")
        numDelet = 0
        while(True):
            filename = open(arr[(count - 1) % len(arr)], "r")   #get file path of each class in order
            lines = filename.readlines()
            if(len(lines) <= 1):
                arr.remove(lines)
                numDelet = numDelet + 1
            else:
                with open(arr[(count - 1) % len(arr)], "w") as f:
                    f.writelines(lines[1:])     # start writing lines except the first line , # lines[1:] from line 2 to last 
                img, mask = get_img_and_mask(lines[0])  # read the first image 
                resize_img(img, bg, mask)    #check boundary of img and scale it into background region

                img0_cor, img1_cor = img.shape[:2]
                x, y = random_coordinate(img, mask, bg, 1, opt.ovl_threshold)
                if(x > 1): # x == -1 means there is no place in bg can hold object with overlapping threshold < 0.5
                    bg[y: y + img0_cor, x: x + img1_cor][mask == 255] = 255     #assign mask into black bg
                    bg2[y: y + img0_cor, x: x + img1_cor][mask ==255] = img[mask == 255]    #assign object in img into background
            count = count + 1
            if(count + numDelet == opt.img_per_bg):
                count = 0
                break
            
        cv2.imwrite("../data_generate_sample2/bg_" + str(i).zfill(5) + ".jpg", bg2)
        print(i)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    