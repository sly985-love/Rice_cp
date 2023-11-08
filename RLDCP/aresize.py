import os

import cv2
import numpy as np
from pathlib import Path


def Large_Scale_Jittering_3(mask, img):
    h, w, _ = img.shape
    # h = img.shape[0]
    # w = img.shape[1]
    a = 640 / h
    b = 640 / w
    if a <= b and a <= 1:
        rescale_ratio = a
    elif a > b and b <= 1:
        rescale_ratio = b
    elif a <= b and a > 1:
        rescale_ratio = a
    elif a > b and b > 1:
        rescale_ratio = b
    else:
        rescale_ratio = 1

    # rescale(等比例缩放，将最大边缩成300)
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

    # crop or padding（然后填充）
    x = 0
    y = 0
    h1 = 640
    w1 = 640
    img_pad = np.ones((h1, w1, 3), dtype=np.uint8) * 168
    mask_pad = np.zeros((h1, w1, 3), dtype=np.uint8)
    img_pad[y:y + h_new, x:x + w_new, :] = img
    mask_pad[y:y + h_new, x:x + w_new, :] = mask
    return mask_pad, img_pad


# data_new_all/Healthy_Pure_532

root = Path('data_new_all/BacterailBlight_Pure_104/bacterialBlight_64')
os.mkdir(root / "JPEGImages")
os.mkdir(root / "SegmentationClass")
root_img = root / "img"
root_mask = root / "mask"
save_img = root / "JPEGImages"
save_mask = root / "SegmentationClass"
orig_list = [path.stem for path in root_img.iterdir()]
mask_list = [path.stem for path in root_mask.iterdir()]
for name in orig_list:
    if name in mask_list:
        path_orig = root_img / (name + '.jpg')
        path_mask = root_mask / (name + '.png')
        mask_src = cv2.imread(str(path_mask))
        img_src = cv2.imread(str(path_orig))
        # print(path_orig, path_mask)
        mask_crop, img_crop = Large_Scale_Jittering_3(mask_src, img_src)
        cv2.imwrite(str(save_mask / (path_orig.stem + '.png')), mask_crop)
        cv2.imwrite(str(save_img / (path_orig.stem + '.jpg')), img_crop)

# BacterailBlight_Pure_104
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BacterailBlight_Pure_104\bacterialBlight_40
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BacterailBlight_Pure_104\bacterialBlight_64
# ___________________________________________
# BacterialBlight_Complex_75
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BacterialBlight_Complex_75\bacterialBlight_19
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BacterialBlight_Complex_75\bacterialBlight_56
# ___________________________________________
# Blast_Complex_27
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Blast_Complex_27\img
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Blast_Complex_27\label
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Blast_Complex_27\mask
# ___________________________________________
# Blast_Pure_364
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Blast_Pure_364\Thumbs.db
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Blast_Pure_364\blast_136
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Blast_Pure_364\blast_188
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Blast_Pure_364\blast_40
# ___________________________________________
# BrownSpot_Complex_42
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BrownSpot_Complex_42\Thumbs.db
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BrownSpot_Complex_42\img
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BrownSpot_Complex_42\label
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BrownSpot_Complex_42\mask
# ___________________________________________
# BrownSpot_Pure_139
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BrownSpot_Pure_139\brownSpot_40
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\BrownSpot_Pure_139\brownSpot_99
# ___________________________________________
# Healthy_Complex_260
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Healthy_Complex_260\img
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Healthy_Complex_260\label
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Healthy_Complex_260\mask
# ___________________________________________
# Healthy_Pure_532
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Healthy_Pure_532\Thumbs.db
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Healthy_Pure_532\img
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Healthy_Pure_532\label
# D:\帅璐宇\Rice\数据集\新收集并标注后的数据集\Healthy_Pure_532\mask
# ___________________________________________
#


# os.mkdir()
