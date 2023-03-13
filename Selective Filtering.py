import cv2
import numpy as np

# Load image
img = cv2.imread('gambar2.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create selective filter
selective_filter = np.zeros_like(gray)
selective_filter[200:400, 200:400] = 1
selective_filter[450:600, 450:600] = 1

# Apply selective filter to image
gray_filtered = gray * selective_filter

# Show filtered image
cv2.imshow('Gambar Original', gray)
cv2.imshow('Selective Filtering', gray_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
