import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar
img = cv2.imread('gambar2.jpg', 0)

# Melakukan transformasi Fourier
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Membuat filter high-pass
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# Menerapkan filter pada gambar
fshift_filtered = fshift * mask
f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_filtered).real

# Menampilkan gambar asli dan hasil filter
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Gambar Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Hasil Filter'), plt.xticks([]), plt.yticks([])
plt.show()
