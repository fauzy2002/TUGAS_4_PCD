import numpy as np
import matplotlib.pyplot as plt

# Membuat gambar berukuran 64x64 piksel dengan titik tengah berwarna putih
image = np.zeros((64, 64))
image[32, 32] = 1

# Melakukan DFT pada gambar
F = np.fft.fft2(image)

# Membuat filter ideal lowpass dengan radius d0=8
M, N = F.shape
u = np.arange(0, M, 1)
v = np.arange(0, N, 1)
idx = np.where(u > M/2)
u[idx] = u[idx] - M
idy = np.where(v > N/2)
v[idy] = v[idy] - N
V, U = np.meshgrid(v, u)
D = np.sqrt(U**2 + V**2)
H = np.zeros((M, N))
H[np.where(D <= 8)] = 1

# Menerapkan filter ideal lowpass pada DFT gambar
G = F * H

# Melakukan inverse DFT pada gambar hasil filter
g = np.real(np.fft.ifft2(G))

# Menampilkan gambar input dan gambar hasil filter
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Gambar Input')
axs[1].imshow(g, cmap='gray')
axs[1].set_title('Gambar Hasil Filter')
plt.show()
