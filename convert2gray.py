import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch

# 将3通道的彩色mask转换成8位的灰度图（注意每个值代表每一类）
label = {'background': 0, 'leaf': 1, 'angular': 2, 'mildew': 3}
path_root = Path(r'D:\SegmentationClass')
path_list = [path for path in path_root.iterdir()]
for jpath in path_list:
    if jpath.suffix == '.png':
        img = Image.open(jpath)
        img = img.convert('L')
        w, h = img.size
        img = np.array(img)
        for j in range(w):
            for i in range(h):
                if img[i][j] == 0:
                    continue
                elif img[i][j] == 100:
                    img[i][j] = label['leaf']
                elif img[i][j] == 51:
                    img[i][j] = label['angular']
                elif img[i][j] == 65:
                    img[i][j] = label['mildew']
                # elif img[i][j] == 149:
                #     img[i][j] = label[name]
        img_pil = Image.fromarray(img, mode='L')
        img_pil.save(jpath)
        print(jpath)
