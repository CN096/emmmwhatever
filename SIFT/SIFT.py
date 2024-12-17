import cv2
import matplotlib.pyplot as plt

# 读取两幅图像
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 在每幅图像上检测SIFT特征点和计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 使用Brute-Force匹配器进行特征点匹配
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 按照距离排序匹配结果
matches = sorted(matches, key=lambda x: x.distance)

# 绘制匹配结果
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示匹配结果
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.axis('off')
plt.show()
