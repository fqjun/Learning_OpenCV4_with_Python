import cv2

import os
dirname, filename = os.path.split(os.path.abspath( __file__))
os.chdir(dirname)
img = cv2.imread("logo.png")
cv2.imshow("Image",img)


