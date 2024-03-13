import os
import re
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


idx=0
# load frames
img = cv2.imread('frame1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.dilate(img, kernel=np.ones((2, 2), np.uint8))

canny = cv2.Canny(gray, 150, 200)
stencil = np.zeros_like(canny)

# specify coordinates of the polygon
polygon = np.array([[0,720],[0,600], [600,240], [720,240], [1280,600],[1280,720]])

# fill polygon with ones
cv2.fillConvexPoly(stencil, polygon, 255)

# plt.figure(figsize=(10,10))
# plt.imshow(stencil, cmap= "gray")
# plt.show()

image = cv2.bitwise_and(canny, canny, mask=stencil)

# plot masked frame
# plt.figure(figsize=(10,10))
# plt.imshow(img, cmap= "gray")
# plt.show()

ret, thresh = cv2.threshold(image, 180, 200, cv2.THRESH_BINARY)

# plot image
# plt.figure(figsize=(10,10))
# plt.imshow(thresh, cmap= "gray")
# plt.show()

lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100,100, maxLineGap=200)

# create a copy of the original frame


# draw Hough lines
for line in lines:
  x1, y1, x2, y2 = line[0]
  cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# plot frame
plt.figure(figsize=(10,10))
plt.imshow(img, cmap= "gray")
plt.show()

