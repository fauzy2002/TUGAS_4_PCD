import cv2
import numpy as np

# Load image
img = cv2.imread('gambar4.jpg')

# Apply median filter with kernel size 3x3
filtered = cv2.medianBlur(img, 3)

# Show filtered image
cv2.imshow('Gambar Original', img)
cv2.imshow('Median Filter', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
