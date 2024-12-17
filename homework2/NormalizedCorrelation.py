import cv2
import numpy as np
import matplotlib.pyplot as plt

# 载入大图和小图
large_image = cv2.imread('CRH2A-2358.jpg')
small_image = cv2.imread('logo.jpg')

# 转换为灰度图
large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

# 使用Normalized Cross-Correlation进行模板匹配
result = cv2.matchTemplate(large_gray, small_gray, cv2.TM_CCOEFF_NORMED)

# 找到最佳匹配位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# 获取小图的尺寸
h, w = small_gray.shape

# 在大图上绘制矩形
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(large_image, top_left, bottom_right, (0, 255, 0), 2)

# 显示大图、小图及响应图
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Large Image')
plt.imshow(cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Small Image')
plt.imshow(cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Response Map')
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.show()
