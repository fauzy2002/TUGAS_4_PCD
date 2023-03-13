import cv2
import numpy as np

# Load image
img = cv2.imread('gambar4.jpg')

# Define kernel size
kernel_size = 3

# Apply min filter using built-in function
filtered = cv2.erode(img, np.ones((kernel_size,kernel_size),np.uint8))

# Show filtered image
cv2.imshow('Gambar Original', img)
cv2.imshow('Min Filter', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
