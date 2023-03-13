import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworth_highpass_filter(image, cutoff, n):
    rows, cols = image.shape
    crow, ccol = int(rows/2), int(cols/2)
    # Create a butterworth highpass filter mask
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d != 0:
                mask[i, j] = 1 / (1 + (cutoff / d)**(2 * n))
    # Apply the butterworth highpass filter mask to the image
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift = fshift * mask
    ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(ishift)
    img_back = np.abs(img_back)
    return img_back

# Load image
img = cv2.imread('gambar2.jpg', 0)

# Apply butterworth highpass filter with cutoff frequency of 30 and order of 3
img_back = butterworth_highpass_filter(img, 30, 3)

# Display original image and filtered image
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(img_back, cmap='gray')
axs[1].set_title('Butterworth Highpass Filtered')
plt.show()
