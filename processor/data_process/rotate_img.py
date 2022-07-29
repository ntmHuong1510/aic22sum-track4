from __future__ import print_function
from ast import arg
from cv2 import rotate
import numpy as np
import imutils
import argparse
import cv2
import random
from random import randint

image = cv2.imread("../../dataset/validate_image/00000_0.jpg") #../dataset/images/00000.jpg



arr = np.arange(0, 360, 15)
angle = arr[np.random.randint(0, len(arr))]
rotated = imutils.rotate(image, angle)
print(angle)
cv2.imshow("Rotated (Problematic)", rotated)
cv2.waitKey(0)
# for angle in np.arange(0, 360, 15):
# 	rotated = imutils.rotate(image, angle)
# 	cv2.imshow("Rotated (Problematic)", rotated)
# 	cv2.waitKey(0)
# # loop over the rotation angles again, this time ensuring
# # no part of the image is cut off
# for angle in np.arange(0, 360, 15):
# 	rotated = imutils.rotate_bound(image, angle)
# 	cv2.imshow("Rotated (Correct)", rotated)
# 	cv2.waitKey(0)