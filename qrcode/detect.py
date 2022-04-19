from re import L
from tkinter import Image
import cv2 
import numpy as np

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop

def find_pts(points):
    x = [points[0][0],points[1][0],points[2][0],points[3][0]]
    y = [points[0][1],points[1][1],points[2][1],points[3][1]]
    # if x[0] < w / 2 and y[0] < h / 2:
    #     return [max(x), max(y)]
    # if x[0] > w / 2 and y[0] < h / 2:
    #     return [min(x), max(y)]
    # if x[0] < w / 2 and y[0] > h / 2:
    #     return [max(x), min(y)]
    # if x[0] > w / 2 and y[0] > h / 2:
    #     return [min(x), min(y)]
    if x[0] < w / 2 and y[0] < h / 2:
        return [max(x), max(y)]
    return [min(x), min(y)]

# image = cv2.imread("test.jpeg")
image = cv2.imread('test.jpeg')
det = cv2.QRCodeDetector()
rv, points = det.detectMulti(image) 
points = points.astype(int)
h = image.shape[0]
w = image.shape[1]
qr1 = find_pts(points[0])
qr2 = find_pts(points[1])
# qr3 = find_pts(points[2])
# qr4 = find_pts(points[3])
# rect = cv2.minAreaRect(np.array([qr1, qr2, qr3, qr4]))
img_croped = image[qr1[0]: qr2[0], qr1[1]: qr2[1]]
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(image, [box], 0, (0, 0, 255), 1)
# cv2.imwrite("result.png",image)
# img_croped = crop_minAreaRect(image, rect)
cv2.imwrite('result.png', img_croped)
