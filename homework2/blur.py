import cv2
import numpy as np
import matplotlib.pyplot as plt

# 载入图像
image = cv2.imread('wrua.jpg')

# 高斯滤波
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# 均值滤波
mean_blur = cv2.blur(image, (5, 5))

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Gaussian Blur')
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Mean Blur')
plt.imshow(cv2.cvtColor(mean_blur, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
