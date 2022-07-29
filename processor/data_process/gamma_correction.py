from __future__ import print_function
from ast import arg
import numpy as np
import argparse
import cv2


# def adjust_gamma(img, gamma = 1.0):
#     #build a lookup table mapping the pixel values [0, 255] to 
#     #their adjustef gamma value
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 250.0) ** invGamma) * 255
#         for i in np.arange(0, 256)])
#     table = np.array(table, np.uint8)
#     #Apply gamma correction using lookup table
#     return cv2.LUT(img, table)

original = cv2.imread("../../dataset/validate_image/00000_0.jpg") #../dataset/images/00000.jpg

# for gamma in np.arange(0.0, 1.2, 0.8):
#     if gamma == 1:
#         continue
#     gamma = gamma if gamma > 0 else 0.1
#     adjusted = adjust_gamma(original, gamma=gamma)
#     cv2.putText(adjusted, "g={}".format(gamma), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 3)

#     cv2.imshow("images", np.hstack([original, adjusted]))
#     cv2.waitKey(0)


def adjust_image_gamma(image, gamma = 1.0):
  image = np.power(image, gamma)
  max_val = np.max(image.ravel())
  image = image/max_val * 255
  image = image.astype(np.uint8)
  return image

img = adjust_image_gamma(original)
cv2.imshow("ori", img)
cv2.waitKey()
# import cv2
# import numpy as np
# import math

# # read image
# img = cv2.imread('index.jpg')

# # METHOD 1: RGB

# # convert img to gray
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # compute gamma = log(mid*255)/log(mean)
# mid = 0.5
# mean = np.mean(gray)
# gamma = math.log(mid*255)/math.log(mean)
# print(gamma)

# # do gamma correction
# img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)



# # METHOD 2: HSV (or other color spaces)

# # convert img to HSV
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# hue, sat, val = cv2.split(hsv)

# # compute gamma = log(mid*255)/log(mean)
# mid = 0.5
# mean = np.mean(val)
# gamma = math.log(mid*255)/math.log(mean)
# print(gamma)

# # do gamma correction on value channel
# val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

# # combine new value channel with original hue and sat channels
# hsv_gamma = cv2.merge([hue, sat, val_gamma])
# img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

# # show results
# cv2.imshow('input', img)
# cv2.imshow('result1', img_gamma1)
# cv2.imshow('result2', img_gamma2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # save results
# cv2.imwrite('lioncuddle1_gamma1.jpg', img_gamma1)
# cv2.imwrite('lioncuddle1_gamma2.jpg', img_gamma2)