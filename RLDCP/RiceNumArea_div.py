import cv2

try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np
from pathlib import Path
import os
# pic_data/BrownSpot/brownSpot_40
# D:\帅璐宇\tools\pic_data\BrownSpot\brownSpot_40  "//" xyp/
# BacterialBlight51  Blast51 BrownSpot96  BB51_10 B51_10 BS96_10  nnndata/Blast_159_1749
root = Path("nnndata""//""Blast_159_1749")
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
        # 按面积比例划分病变等级
        if (k / (m + k) <= 0.1):
            cv2.imwrite(str(save_img_1 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_1 / (path_orig.stem + '.png')), image)
        elif (k / (m + k) > 0.1 and k / (m + k) <= 0.25):
            cv2.imwrite(str(save_img_2 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_2 / (path_orig.stem + '.png')), image)
        elif (k / (m + k) > 0.25 and k / (m + k) <= 0.45):
            cv2.imwrite(str(save_img_3 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_3 / (path_orig.stem + '.png')), image)
        elif (k / (m + k) > 0.45 and k / (m + k) <= 0.65):
            cv2.imwrite(str(save_img_4 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_4 / (path_orig.stem + '.png')), image)
        elif (k / (m + k) > 0.65):
            cv2.imwrite(str(save_img_5 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_5 / (path_orig.stem + '.png')), image)
        else:
            cv2.imwrite(str(save_img_0 / (path_orig.stem + '.jpg')), img_src)
            cv2.imwrite(str(save_mask_0 / (path_orig.stem + '.png')), image)