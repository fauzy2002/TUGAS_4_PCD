import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar
img = cv2.imread('gambar1.jpg', 0)

# Melakukan transformasi Fourier diskret
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# Membuat filter high-pass
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# Menerapkan filter pada gambar
dft_shift_filtered = dft_shift * mask
dft_filtered = np.fft.ifftshift(dft_shift_filtered)
img_filtered = np.fft.ifft2(dft_filtered)
img_filtered = np.real(img_filtered)

# Menampilkan gambar asli dan hasil filter
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Gambar Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Hasil Filter'), plt.xticks([]), plt.yticks([])
plt.show()
