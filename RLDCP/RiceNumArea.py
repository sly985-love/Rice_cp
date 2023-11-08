import cv2

try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np

# 计算病害严重程度
image = cv2.imread('data3/SegmentationClass/DSC_0104.png')
img = cv2.imread('data3/SegmentationClass/DSC_0104.png', 0)
height = image.shape[0]
width = image.shape[1]
k = 0
m = 0
n = 0
# 计算各颜色像素面积
for i in range(height):
    for j in range(width):
        if (image[i, j][0] == 6 and image[i, j][1] == 128 and image[i, j][2] == 245):
            k = k + 1
        elif (image[i, j][0] == 181 and image[i, j][1] == 119 and image[i, j][2] == 53):
            m = m + 1
# 按面积比例划分病变等级
if (k / (m + k) <= 0.1):
    n = 1
elif (k / (m + k) > 0.1 and k / (m + k) <= 0.25):
    n = 2
elif (k / (m + k) > 0.25 and k / (m + k) <= 0.45):
    n = 3
elif (k / (m + k) > 0.45 and k / (m + k) <= 0.65):
    n = 4
elif (k / (m + k) > 0.65):
    n = 5
else:
    n = 0
print("病害像素面积为：", k)
print("叶片像素面积为：", m)
print("疾病严重程度为：", n)
# 计算病害个数
# img = cv2.imread('data3/SegmentationClass/DSC_0100.png', 0)
h = img.shape[0]
w = img.shape[1]
for i in range(h):
    for j in range(w):
        if img[i, j] == 106:
            img[i, j] = 0
        elif img[i, j] == 149:
            img[i, j] = 225
# 轮廓发现
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("病斑个数共计:", len(contours))

# 在图片上显示信息 并画框
dn = len(contours)
q = 0
if dn >= 1 and dn <= 5:
    q = 1
elif dn > 5 and dn <= 10:
    q = 2
elif dn > 10 and dn <= 15:
    q = 3
elif dn > 15 and dn <= 20:
    q = 4
elif dn >20:
    q = 5
else:
    q = 0
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, "disease area:{}".format(k), (5, 370), font, 0.5, (255, 255, 255), 1)
cv2.putText(image, "leaf area:{}".format(m), (5, 400), font, 0.5, (255, 255, 255), 1)
cv2.putText(image, "disease severity by are:{}".format(n), (5, 430), font, 0.5, (255, 255, 255), 1)
cv2.putText(image, "disease number:{}".format(dn), (5, 475), font, 0.5, (255, 255, 255), 1)
cv2.putText(image, "disease severity by number:{}".format(q), (5, 500), font, 0.5, (255, 255, 255), 1)

for c in range(len(contours)):
    rect = cv2.minAreaRect(contours[c])
    cx, cy = rect[0]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    cv2.circle(image, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)
    cv2.drawContours(image, contours, c, (0, 0, 255), 1, 8)

cv2.imshow("reasult", image)
cv2.imwrite("result.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
