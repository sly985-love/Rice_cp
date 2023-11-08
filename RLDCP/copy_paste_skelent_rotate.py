"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

from PIL import Image
# import imgviz
import cv2
import argparse
import os
import numpy as np
import tqdm
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# 图片细化（骨架提取）多张图片批量处理
import cv2
from skimage import morphology
import numpy as np
import glob
import os

imageList = glob.glob("imgs/*.png")  # 读取原始图片
img_file = "data/"  # 保存骨架化后的图片的地址
# 确认上述地址是否存在
if not os.path.exists(img_file):
    os.mkdir(img_file)

# 批量处理图片
for item in imageList:
    # print(item)           # imgs/293718492401.pdf10.png
    img = cv2.imread(item, 0)  # 导入图片
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  # 二值化处理
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)  # 图片细化（骨架提取）
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8) * 255
    cv2.imwrite(img_file + item[5:], dist_on_skel)  # 此处dist_on_skel就是你需要保存的图像文件


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img


def rotate(mask_src, mask0_src, img_src):
    # 根据mask0_src获取需要旋转的度数
    h = mask0_src.shape[0]
    w = mask0_src.shape[1]
    for i in range(h):
        for j in range(w):
            # print(mask0_src[i, j])
            if mask0_src[i, j] == 106 or mask0_src[i, j] == 149:
                mask0_src[i, j] = 255
    # 再闭操作一次：去掉目标特征内的孔
    kernel = np.ones((3, 3), np.uint8)
    mask0_src = cv2.morphologyEx(mask0_src, cv2.MORPH_CLOSE, kernel)
    # 检测边缘
    t = 100
    mask0_src = cv2.Canny(mask0_src, t, t * 2)
    k = np.ones((3, 3), dtype=np.uint8)
    mask0_src = cv2.morphologyEx(mask0_src, cv2.MORPH_DILATE, k)
    # 轮廓发现
    contours, hierarchy = cv2.findContours(mask0_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b = 1000
    for c in range(len(contours)):
        rect = cv2.minAreaRect(contours[c])  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        a = rect[2]
        if a <= b:
            b = a
    # 根据获取的旋转角度旋转掩码图和原图
    rows, cols, channel = mask_src.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), b - 90, 1)
    mask_src = cv2.warpAffine(mask_src, M, (cols, rows))
    img_src = cv2.warpAffine(img_src, M, (cols, rows))
    return mask_src, img_src


def Middle_Scale_Jittering_3(mask, img, min_scale=1, max_scale=1.2):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    # h, w, _ = img.shape
    h = img.shape[0]
    w = img.shape[1]

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w, 3), dtype=np.uint8)
        img_pad[y:y + h_new, x:x + w_new, :] = img
        mask_pad[y:y + h_new, x:x + w_new, :] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y + h, x:x + w, :]
        mask_crop = mask[y:y + h, x:x + w, :]
        return mask_crop, img_crop


# 3通道的mask
# ------对mask和对应的原图进行尺度缩放，再对缩放后图像进行填充（缩小了）或裁剪（放大了），保证输出还是原来的大小------#
def Large_Scale_Jittering_3(mask, img, min_scale=0.2, max_scale=2.0):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    # h, w, _ = img.shape
    h = img.shape[0]
    w = img.shape[1]

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w, 3), dtype=np.uint8)
        img_pad[y:y + h_new, x:x + w_new, :] = img
        mask_pad[y:y + h_new, x:x + w_new, :] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y + h, x:x + w, :]
        mask_crop = mask[y:y + h, x:x + w, :]
        return mask_crop, img_crop


def new_img_add(mask_src, img_src, mask_dst, img_dst):
    mask_src1 = mask_src
    img_src1 = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img_dst1 = cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB)
    mask_dst1 = cv2.cvtColor(mask_dst, cv2.COLOR_BGR2RGB)
    # 背景图像的shape（用于粘贴病害）
    height_src = mask_src.shape[0]
    width_src = mask_src.shape[1]
    # 疾病图像的shape（用于扣取病害的mask图）
    height_dst = mask_dst.shape[0]
    width_dst = mask_dst.shape[1]

    # 计算各颜色像素面积
    for i in range(height_dst):
        for j in range(width_dst):
            if (mask_dst[i, j][0] == 6 and mask_dst[i, j][1] == 128 and mask_dst[i, j][2] == 245):
                mask_dst[i, j][0] = 255
                mask_dst[i, j][1] = 255
                mask_dst[i, j][2] = 255
            else:
                mask_dst[i, j][0] = 0
                mask_dst[i, j][1] = 0
                mask_dst[i, j][2] = 0
    # 转换通道

    # vfhgtlhng
    mask_src = cv2.cvtColor(mask_src, cv2.COLOR_BGR2RGB)
    mask_dst = cv2.cvtColor(mask_dst, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask_dst, cv2.COLOR_BGR2GRAY)
    mask = mask.astype('uint8')
    center = [0, 0]  # 在新背景图片中的位置
    for i in range(height_src):
        for j in range(width_src):
            if mask[i, j] == 255 and mask_src[i, j][0] != 0 and mask_src[i, j][1] != 0 and mask_src[i, j][
                2] != 0:  # 0代表黑色的点
                mask_src[center[0] + i, center[1] + j] = mask_dst1[i, j]  # 此处替换颜色，为BGR通道
                img_src1[center[0] + i, center[1] + j] = img_dst1[i, j]
    new_mask = cv2.cvtColor(mask_src, cv2.COLOR_RGB2BGR)
    new_img = cv2.cvtColor(img_src1, cv2.COLOR_RGB2BGR)
    return new_mask, new_img


def new_copy_paste(mask_src, img_src, mask_dst, img_dst):
    # # （1）翻转
    # mask_src, img_src = random_flip_horizontal(mask_src, img_src)  # 根据随机概率判断是否做翻转
    # mask_dst, img_dst = random_flip_horizontal(mask_dst, img_dst)  # 根据随机概率判断是否做翻转
    # （2）大尺度抖动 LSJ， Large_Scale_Jittering
    mask_src, img_src = Middle_Scale_Jittering_3(mask_src, img_src)  # 做尺度缩放、裁剪或填充，保证输出和原来一样大小
    # mask_dst, img_dst = Large_Scale_Jittering_3(mask_dst, img_dst)
    # （3）抠图与粘图
    mask, img = new_img_add(mask_src, img_src, mask_dst, img_dst)
    mask, img = Large_Scale_Jittering_3(mask, img)
    # 随机旋转
    return mask, img


def save_img(image, path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i += 1
        print(path_pattern % i)
    image.save(path_pattern % i)

    # cv2.imwrite(path_pattern % i)


def main():
    # root = Path('data2')
    # root_img = root / "JPEGImages"
    # root_mask = root / "SegmentationClass"
    # save_img = root / "Bacterialblight" / "JPEGImages"
    # save_mask = root / "Bacterialblight" / "SegmentationClass"
    root = Path("pic_data" "//" "Blast")
    # os.mkdir(root / "Blast39_10" / "JPEGImages")
    # os.mkdir(root / "Blast39_10" / "SegmentationClass")
    root_img = root / "blast_39-" / "JPEGImages"
    root_mask = root / "blast_39-" / "SegmentationClass"
    save_img = root / "Blast39_10" / "JPEGImages"
    save_mask = root / "Blast39_10" / "SegmentationClass"
    orig_list = [path.stem for path in root_img.iterdir()]
    mask_list = [path.stem for path in root_mask.iterdir()]

    for name in orig_list:
        if name in mask_list:
            path_orig = root_img / (name + '.jpg')
            path_mask = root_mask / (name + '.png')
            # mask_list.remove(name)
            mask_dst_path_ = np.random.choice(mask_list)  # 随机选取mask里面一张
            mask_dst_path = root_mask / (mask_dst_path_ + '.png')
            img_dst_path = root_img / (mask_dst_path_ + '.jpg')
            # print("选着了图片",mask_dst_path)
            img_src = cv2.imread(str(path_orig))
            mask_src = cv2.imread(str(path_mask))
            mask0_src = cv2.imread(str(path_mask), 0)
            mask_dst = cv2.imread(str(mask_dst_path))
            img_dst = cv2.imread(str(img_dst_path))
            mask0_dst = cv2.imread(str(mask_dst_path), 0)
            # 图像先旋转
            mask_src, img_src = rotate(mask_src, mask0_src, img_src)
            mask_dst, img_dst = rotate(mask_dst, mask0_dst, img_dst)
            M = 1
            for i in range(8,9):
                new_mask, new_img = new_copy_paste(mask_src, img_src, mask_dst, img_dst)
                cv2.imwrite(str(save_img / (path_orig.stem + "_" + str(i) + '.jpg')), new_img)
                cv2.imwrite(str(save_mask / (path_orig.stem + "_" + str(i) + '.png')), new_mask)


if __name__ == '__main__':
    main()
