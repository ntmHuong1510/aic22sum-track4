"""
    Test the RoI in background with openCV

    bg_roi = [  [492, 247, 758, 600],
                [582, 292, 786, 621],   
                [599, 303, 765, 606],
                [589, 304, 775, 602],
                [602, 316, 777, 594]
            ]
"""
import cv2
import numpy as np

# img = cv2.imread("../../background/testA_1/1.jpg")
img = cv2.imread("../../frame_extract/testA_5/00020.jpg")
# processor/frame_extract/testA_2/00000.jpg
x, y, w, h = cv2.selectROI(img)
cv2.destroyAllWindows()
bg = np.zeros((img.shape[0],img.shape[1],3), np.uint8) #height, width, channel

bg[y: y + h, x: x + w] = 255
cv2.imshow("ROI", bg)
cv2.waitKey()
cv2.imwrite("RoI_bg.png", bg)
print(x, y, w, h)

