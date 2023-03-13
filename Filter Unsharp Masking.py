import cv2
import numpy as np
import matplotlib.pyplot as plt

def unsharp_masking(image, kernel_size, alpha):
    # Create a gaussian blur filter mask
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # Subtract the blurred image from the original image to create a mask
    mask = image - blur
    # Add the mask back to the original image with an alpha value
    img_back = image + alpha * mask
    return img_back

# Load image
img = cv2.imread('gambar2.jpg', 0)

# Apply unsharp masking with kernel size of 5 and alpha value of 1.5
img_back = unsharp_masking(img, 5, 1.5)

# Display original image and filtered image
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(img_back, cmap='gray')
axs[1].set_title('Unsharp Masking Filtered Image')
plt.show()
