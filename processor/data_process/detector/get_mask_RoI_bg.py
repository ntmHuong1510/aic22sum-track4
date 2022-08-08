"""
    Test the RoI in background with openCV
"""
import cv2
import numpy as np

img = cv2.imread("../../background/testA_1/1.jpg")
x, y, w, h = cv2.selectROI(img)
cv2.destroyAllWindows()
bg = np.zeros((img.shape[0],img.shape[1],3), np.uint8) #height, width, channel

bg[y: y + h, x: x + w] = 255
filelabel = open("coordinate.txt", "w")
cv2.imshow("ROI", bg)
cv2.waitKey()
cv2.imwrite("RoI_bg.png", bg)
print(x, y, w, h)

