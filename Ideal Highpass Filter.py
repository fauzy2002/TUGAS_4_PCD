import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load image
img = mpimg.imread('gambar2.jpg')

# Convert to grayscale
gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])

# Perform Fourier Transform
f = fftpack.fft2(gray)
fshift = fftpack.fftshift(f)

# Define highpass filter mask
rows, cols = gray.shape
crow, ccol = int(rows/2), int(cols/2)
mask = np.ones((rows, cols))
r = 50
mask[crow-r:crow+r, ccol-r:ccol+r] = 0

# Apply highpass filter mask
fshift = fshift * mask

# Perform Inverse Fourier Transform
ishift = fftpack.ifftshift(fshift)
img_back = fftpack.ifft2(ishift)
img_back = np.abs(img_back)

# Display original image and filtered image
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(gray, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(img_back, cmap='gray')
axs[1].set_title('Ideal Highpass Filtered')
plt.show()
