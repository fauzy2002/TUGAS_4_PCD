import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread('gambar2.jpg', 0)

# Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Construct Laplacian filter
rows, cols = img.shape
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], np.float32)

# Apply Laplacian filter
laplacian_kernel = np.fft.fft2(laplacian_kernel, s=(rows, cols))
laplacian_kernel = np.fft.fftshift(laplacian_kernel)
laplacian_filtered = np.real(np.fft.ifft2(fshift * laplacian_kernel))

# Display original and filtered image
cv2.imshow('Original Image', img)
cv2.imshow('Laplacian Filtered Image', laplacian_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
