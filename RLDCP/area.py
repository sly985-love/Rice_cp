import cv2

image = cv2.imread("data/SegmentationClass/brownspot_orig_100.png")  # 输入二值图像
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
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(image, "disease area:{}".format(k), (5, 15), font, 0.5, (255, 255, 255), 1)
img = cv2.putText(image, "leaf area:{}".format(m), (5, 45), font, 0.5, (255, 255, 255), 1)
img = cv2.putText(image, "disease severity:{}".format(n), (5, 75), font, 0.5, (255, 255, 255), 1)
cv2.imwrite('result.png',img)
cv2.imshow("res", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
