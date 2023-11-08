import cv2

try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np
from pathlib import Path
import os

# pic_data/BrownSpot/brownSpot_40
# D:\帅璐宇\tools\pic_data\BrownSpot\brownSpot_40  "//" xyp/  nnndata/BrownSpot_135_1485
# BacterialBlight51  Blast51 BrownSpot96  BB51_10 B51_10 BS96_10
root = Path("nnndata""//""BrownSpot_90-_990")
os.mkdir(root / "Severity_0" / "JPEGImages")
os.mkdir(root / "Severity_0" / "SegmentationClass")
os.mkdir(root / "Severity_1" / "JPEGImages")
os.mkdir(root / "Severity_1" / "SegmentationClass")
os.mkdir(root / "Severity_2" / "JPEGImages")
os.mkdir(root / "Severity_2" / "SegmentationClass")
os.mkdir(root / "Severity_3" / "JPEGImages")
os.mkdir(root / "Severity_3" / "SegmentationClass")
os.mkdir(root / "Severity_4" / "JPEGImages")
os.mkdir(root / "Severity_4" / "SegmentationClass")
os.mkdir(root / "Severity_5" / "JPEGImages")
os.mkdir(root / "Severity_5" / "SegmentationClass")
root_img = root / "JPEGImages"
root_mask = root / "SegmentationClass"
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
save_img_0 = root / "Severity_5" / "JPEGImages"
save_mask_0 = root / "Severity_5" / "SegmentationClass"
orig_list = [path.stem for path in root_img.iterdir()]
mask_list = [path.stem for path in root_mask.iterdir()]
# list_num = []
for name in orig_list:
    if name in mask_list:
        path_orig = root_img / (name + '.jpg')
        path_mask = root_mask / (name + '.png')
        img_src = cv2.imread(str(path_orig))
        image = cv2.imread(str(path_mask))
        img = cv2.imread(str(path_mask), 0)

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
        q = 0
        if dn >= 1 and dn <= 5:
             cv2.imwrite(str(save_img_1 / (path_orig.stem + '.jpg')), img_src)
             cv2.imwrite(str(save_mask_1 / (path_orig.stem + '.png')), image)
        elif dn > 5 and dn <= 10:
             cv2.imwrite(str(save_img_2 / (path_orig.stem + '.jpg')), img_src)
             cv2.imwrite(str(save_mask_2 / (path_orig.stem + '.png')), image)
        elif dn > 10 and dn <= 15:
             cv2.imwrite(str(save_img_3 / (path_orig.stem + '.jpg')), img_src)
             cv2.imwrite(str(save_mask_3 / (path_orig.stem + '.png')), image)
        elif dn > 15 and dn <= 20:
             cv2.imwrite(str(save_img_4 / (path_orig.stem + '.jpg')), img_src)
             cv2.imwrite(str(save_mask_4 / (path_orig.stem + '.png')), image)
        elif dn > 20:
             cv2.imwrite(str(save_img_5 / (path_orig.stem + '.jpg')), img_src)
             cv2.imwrite(str(save_mask_5 / (path_orig.stem + '.png')), image)
        else:
            cv2.imwrite(str(save_img_0 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_0 / (path_orig.stem + '.png')), image)
        # list_num.append(len(contours))
# print(list_num)
# a = [4, 3, 16, 9, 4, 2, 3, 6, 17, 16, 1, 5, 2, 3, 2, 7, 2, 0, 2, 1, 4, 2, 8, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 5, 1, 3,
#      1, 13, 1, 3, 3, 1, 1, 4, 3, 2, 2, 3, 4, 1, 1, 2, 3, 3, 1, 2, 3, 6, 1, 2, 3, 2, 1, 1, 6, 2, 2, 4, 2, 1, 7, 3, 5, 2,
#      2, 1, 1, 3, 1, 9, 6, 2, 2, 4, 1, 2, 1, 1, 3, 1, 2, 1, 43, 27]
# a.sort()
# print(a)
# b = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
#      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
#      4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 9, 13, 16, 16, 17, 27, 43]
