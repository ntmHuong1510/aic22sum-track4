import cv2
import numpy as np
import math


img2 = cv2.imread('../../dataset/validate_image/00115_116374.jpg') #dataset\validate_image\00000_0.jpg
mask = cv2.imread('../../dataset/segmentation_labels/00116_116375_seg.jpg')
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

mask[np.bitwise_and(mask < 255, mask > 200)] = 255
mean = np.sum(np.minimum(mask, img2)) / np.count_nonzero(mask)

hue, sat, val = cv2.split(hsv)

mid = 0.5
gamma = math.log(mid*255)/math.log(mean)
print(gamma)

# do gamma correction on value channel
val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

# combine new value channel with original hue and sat channels
hsv_gamma = cv2.merge([hue, sat, val_gamma])
img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

cv2.imshow('result1', img_gamma2)
cv2.imshow('input', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
