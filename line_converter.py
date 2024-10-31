import cv2
import numpy as np

# Covert image to grayscale
image = cv2.imread("rotated_2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Using Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Using HoughLinesP to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 0), 2)

# Save image
cv2.imwrite("line.jpg", gray)