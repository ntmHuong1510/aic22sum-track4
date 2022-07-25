from os import listdir, path, rename
import random
from random import randint
from linereader import copen
import numpy as np
import cv2

if __name__ == '__main__':
# So luong obj trong bg la random, con so luong bg  
    file = copen("linkpath_image_segment.txt")
    lines = file.count('\n')
    bg = cv2.imread('background.jpg')
    print("Enter number of image need to random: ")
    n = int(input())    

    for i in range(n):
        #Read randome line in file contains relative path of images and corresponding segmentation
        # format int path file: class_id:Link_img Link_segment
        random_line = file.getline(randint(1, lines))   
        list = random_line.split(" ")
        mask = cv2.imread(list[1].split('\n')[0])   # Read segment of img
        img = cv2.imread(list[0].split(':')[1])     #Read img
        img0_cor, img1_cor = img.shape[:2]
        x, y = np.random.randint(0, bg.shape[1] - img1_cor), np.random.randint(0, bg.shape[0] - img0_cor)
        bg[y: y + img0_cor, x: x + img1_cor][mask > 0] = img[mask > 0]
        # cv2.imwrite("../imgGenerator", bg)
    cv2.imshow("Bg", bg)
    cv2.waitKey()