import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import segment_reduce


bg = cv2.imread('background.jpg')

rel_path_segLabel = '../../dataset/segmentation_labels/'
rel_path_Img = '../../dataset/validate_image/'
mask = cv2.imread(rel_path_segLabel + '00002_15188_seg.jpg')
img = cv2.imread(rel_path_Img + '00001_15187.jpg')


mask = (mask * 255).round().astype(np.uint8)
print(mask)




cv2.imshow("original", bg)
cv2.waitKey()

