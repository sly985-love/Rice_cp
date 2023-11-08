import cv2

try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np
from pathlib import Path
import os

# pic_data/BrownSpot/brownSpot_40
# D:\帅璐宇\tools\pic_data\BrownSpot\brownSpot_40  "//" xyp/  202207092/bs150
# BacterialBlight51  Blast51 BrownSpot96  BB51_10 B51_10 BS96_10  nnndata/Blast_159_1749 BacterialBlight_154_1630
root = Path(r"E:\data\ricedata_v3.0\bs150")
for i in range(5):
    pathname = 'Severity_%d'.format(i)
    if not (root / pathname).exists():
        os.mkdir(root / pathname)
        os.mkdir(root / pathname / "JPEGImages")
        os.mkdir(root / pathname / "SegmentationClass")
    elif not (root / pathname / "JPEGImages").exists():
        os.mkdir(root / pathname / "JPEGImages")
    elif not (root / pathname / "SegmentationClass").exists():
        os.mkdir(root / pathname / "SegmentationClass")
# os.mkdir(root / "Severity_1" / "JPEGImages")
# os.mkdir(root / "Severity_1" / "SegmentationClass")
# os.mkdir(root / "Severity_2" / "JPEGImages")
# os.mkdir(root / "Severity_2" / "SegmentationClass")
# os.mkdir(root / "Severity_3" / "JPEGImages")
# os.mkdir(root / "Severity_3" / "SegmentationClass")
# os.mkdir(root / "Severity_4" / "JPEGImages")
# os.mkdir(root / "Severity_4" / "SegmentationClass")
# os.mkdir(root / "Severity_5" / "JPEGImages")
# os.mkdir(root / "Severity_5" / "SegmentationClass")
root_img = root / "JPEGImages"
root_mask = root / "SegmentationClass"
save_img_0 = root / "Severity_0" / "JPEGImages"
save_mask_0 = root / "Severity_0" / "SegmentationClass"
save_img_1 = root / "Severity_1" / "JPEGImages"
save_mask_1 = root / "Severity_1" / "SegmentationClass"
save_img_2 = root / "Severity_2" / "JPEGImages"
save_mask_2 = root / "Severity_2" / "SegmentationClass"
save_img_3 = root / "Severity_3" / "JPEGImages"
save_mask_3 = root / "Severity_3" / "SegmentationClass"
save_img_4 = root / "Severity_4" / "JPEGImages"
save_mask_4 = root / "Severity_4" / "SegmentationClass"
save_img_5 = root / "Severity_5" / "JPEGImages"
save_mask_5 = root / "Severity_5" / "SegmentationClass"
orig_list = [path.stem for path in root_img.iterdir()]
mask_list = [path.stem for path in root_mask.iterdir()]
leavelnum = [0,0,0,0,0]
for name in orig_list:
    if name in mask_list:
        path_orig = root_img / (name + '.jpg')
        path_mask = root_mask / (name + '.png')
        img_src = cv2.imread(str(path_orig))
        image = cv2.imread(str(path_mask))
        img = cv2.imread(str(path_mask), 0)
        height = image.shape[0]
        width = image.shape[1]
        k = 0
        m = 0
        
        # 计算各颜色像素面积
        for i in range(height):
            for j in range(width):
                if (image[i, j][0] == 6 and image[i, j][1] == 128 and image[i, j][2] == 245):
                    k = k + 1
                elif (image[i, j][0] == 181 and image[i, j][1] == 119 and image[i, j][2] == 53):
                    m = m + 1
        n = 0
        a = 0
        b = 0
        # 按面积比例划分病变等级
        if (k == 0):
            a = 0
            
        elif (k / (m + k) <= 0.1):
            a = 1
        elif (k / (m + k) > 0.1 and k / (m + k) <= 0.25):
            a = 2
        elif (k / (m + k) > 0.25 and k / (m + k) <= 0.45):
            a = 3
        elif (k / (m + k) > 0.45 and k / (m + k) <= 0.65):
            a = 4
        elif (k / (m + k) > 0.65):
            a = 5
        else:
            pass
        # img = cv2.imread(str(path_mask), 0)

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
        # 在图片上显示信息 并画框
        dn = len(contours)
        if dn >= 1 and dn <= 5:
            b = 1
        elif dn > 5 and dn <= 10:
            b = 2
        elif dn > 10 and dn <= 15:
            b = 3
        elif dn > 15 and dn <= 20:
            b = 4
        elif dn > 20:
            b = 5
        else:
            b = 0
        c = 0
        if (a >= b):
            c = a
        else:
            c = b
        leavelnum[c] += 1
        if (c == 1):
            cv2.imwrite(str(save_img_1 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_1 / (path_orig.stem + '.png')), image)
        elif (c == 2):
            cv2.imwrite(str(save_img_2 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_2 / (path_orig.stem + '.png')), image)
        elif (c == 3):
            cv2.imwrite(str(save_img_3 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_3 / (path_orig.stem + '.png')), image)
        elif (c == 4):
            cv2.imwrite(str(save_img_4 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_4 / (path_orig.stem + '.png')), image)
        elif (c == 5):
            cv2.imwrite(str(save_img_5 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_5 / (path_orig.stem + '.png')), image)
        else:
            cv2.imwrite(str(save_img_0 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_0 / (path_orig.stem + '.png')), image)
print(leavelnum)
