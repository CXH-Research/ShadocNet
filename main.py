import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img = cv.imread('./sample/1-1.png')
# Gus0 = img
# Gus1 = cv.pyrDown(Gus0)
# Gus2 = cv.pyrDown(Gus1)
# Gus3 = cv.pyrDown(Gus2)
#
# Lap0 = Gus0 - cv.pyrUp(Gus1)
# Lap1 = Gus1 - cv.pyrUp(Gus2)
# Lap2 = Gus2 - cv.pyrUp(Gus3)
#
# prGus = cv.pyrUp(Gus3)
# for i in range(2):
#     prGus = cv.pyrUp(prGus)
#
# plt.figure(0)
# plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Org/Gus0'), plt.axis('off')
# plt.subplot(222), plt.imshow(Gus3, 'gray'), plt.title('Gus3'), plt.axis('off')  # 为了效果明显 我们选用第3层高斯
# plt.subplot(223), plt.imshow(cv.pyrUp(Gus3), 'gray'), plt.title('prGus'), plt.axis('off')  # 如果我们直接上采样
# plt.subplot(224), plt.imshow(Lap2, 'gray'), plt.title('LAP2'), plt.axis('off')
#
# plt.figure(1)
# rep = Lap0 + cv.pyrUp(Lap1 + cv.pyrUp(Lap2 + cv.pyrUp(Gus3)))
# plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Org/Gus0'), plt.axis('off')
# plt.subplot(122), plt.imshow(rep, 'gray'), plt.title('LapToRestore'), plt.axis('off')
# plt.show()

