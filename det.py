
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1) Load image
img = cv2.imread("./frame.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2) Gray Scale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# gray_img = cv2.dilate(gray_img, kernel=np.ones((2, 2), np.uint8))
# gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
# plt.imshow(gray_img)
# plt.show()
# Step 3) Canny
canny = cv2.Canny(gray_img, 150, 300)
# plt.imshow(canny)
# plt.show()
# Step 4) define ROI Vertices
roi_vertices = np.array([[0,720],[0,600], [600,240], [720,240], [1280,600],[1280,720]])


# Step 5) define ROI function
def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


# Step 6) ROI Image
roi_image = roi(canny, np.array([roi_vertices], np.int32))
# plt.imshow(roi_image)
# plt.show()
# Step 7) Apply Hough Lines P Method on ROI Image
lines = cv2.HoughLinesP(roi_image, 1, np.pi/180, 100, minLineLength=100, maxLineGap=1000)

print(lines,lines.shape)
# Step 8) Draw Hough lines
def draw_lines(image, hough_lines):
    slopes=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


final_img = draw_lines(img, lines)  # Result

plt.imshow(final_img)
plt.xticks([])
plt.yticks([])
plt.show()