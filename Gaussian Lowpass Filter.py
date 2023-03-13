import cv2
import numpy as np

# Fungsi untuk membuat filter Gaussian
def create_gaussian_kernel(ksize, sigma):
    # Membuat kernel kosong
    kernel = np.zeros((ksize, ksize))

    # Hitung nilai kernel berdasarkan rumus Gaussian
    for i in range(ksize):
        for j in range(ksize):
            x = i - ksize // 2
            y = j - ksize // 2
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / (2 * np.pi * sigma**2)

    return kernel

# Fungsi untuk menerapkan filter Gaussian pada gambar
def apply_gaussian_filter(img, ksize, sigma):
    # Membuat kernel Gaussian
    kernel = create_gaussian_kernel(ksize, sigma)

    # Menggunakan filter konvolusi 2D dari OpenCV
    filtered_img = cv2.filter2D(img, -1, kernel)

    return filtered_img

# Membaca gambar
img = cv2.imread('gambar3.png', cv2.IMREAD_GRAYSCALE)

# Menerapkan filter Gaussian dengan kernel 5x5 dan sigma = 1
filtered_img = apply_gaussian_filter(img, ksize=5, sigma=1)

# Menampilkan gambar asli dan hasil filter
cv2.imshow('Gambar asli', img)
cv2.imshow('Hasil filter', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
