import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import segment_reduce



bg = cv2.imread('background.jpg')

rel_path_segLabel = '../../Dataset/segmentation_labels/'
rel_path_Img = '../../Dataset/validateImage/'
mask = [cv2.imread(rel_path_segLabel + '00002_15188_seg.jpg'), cv2.imread(rel_path_segLabel + '00003_101522_seg.jpg'), cv2.imread(rel_path_segLabel + '00004_114967_seg.jpg'), cv2.imread(rel_path_segLabel + '00005_15423_seg.jpg')]
img = [cv2.imread(rel_path_Img + '00001_15187.jpg'), cv2.imread(rel_path_Img + '00002_101521.jpg'), cv2.imread(rel_path_Img + '00003_114966.jpg'), cv2.imread(rel_path_Img + '00004_15422.jpg')]

# cv2.imshow("or", cv2.bitwise_and(mask, img))



# for i in range(4):
#     img0_cor, img1_cor = img[i].shape[:2]
#     x, y = np.random.randint(0, bg.shape[1] - img0_cor), np.random.randint(0, bg.shape[0] - img1_cor)
#     bg[y: y + img0_cor, x: x + img1_cor][mask[i] > 0] = img[i][mask[i] > 0]
#     # crop_img = bg[y: y + img[i].shape[0], x: x + img[i].shape[1]]
#     # crop_img[mask[i] > 0] = img[i][mask[i] > 0]
 
#     # h, w = crop_img.shape[:2]
#     # xx, yy = np.random.randint(0, bg.shape[1] - w), np.random.randint(0, bg.shape[0] - h)
#     # # result = bg.copy()
#     # bg[yy:yy + h, xx: xx + w] = crop_img
#     # cv2.imwrite("backgroundTest.jpg", bg)

cv2.imshow("original", bg)
cv2.waitKey()

