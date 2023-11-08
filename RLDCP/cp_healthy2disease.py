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


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img


# 3通道的mask
# ------对mask和对应的原图进行尺度缩放，再对缩放后图像进行填充（缩小了）或裁剪（放大了），保证输出还是原来的大小------#
def Large_Scale_Jittering_3(mask, img, min_scale=0.1, max_scale=2.0):
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
    # （1）翻转
    # mask_src, img_src = random_flip_horizontal(mask_src, img_src)  # 根据随机概率判断是否做翻转
    # mask_dst, img_dst = random_flip_horizontal(mask_dst, img_dst)  # 根据随机概率判断是否做翻转
    # （2）大尺度抖动 LSJ， Large_Scale_Jittering
    # mask_src, img_src = Large_Scale_Jittering_3(mask_src, img_src)  # 做尺度缩放、裁剪或填充，保证输出和原来一样大小
    # mask_dst, img_dst = Large_Scale_Jittering_3(mask_dst, img_dst)
    # （3）抠图与粘图
    mask, img = new_img_add(mask_src, img_src, mask_dst, img_dst)
    return mask, img


def save_img(image, path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i += 1
        print(path_pattern % i)
    image.save(path_pattern % i)

    # cv2.imwrite(path_pattern % i)

# data_new_all/BacterialBlight_Pure_104/bacterialBlight_40
# data_new_all/BrownSpot_Pure_139/brownSpot_99
def main():
    root = Path('data_new_all')
    os.mkdir(root / "Healthy2BacterialBlight_Pure40" / "JPEGImages")
    os.mkdir(root / "Healthy2BacterialBlight_Pure40" / "SegmentationClass")
    root_img = root / "Healthy_Pure_530" / "JPEGImages"
    root_mask = root / "Healthy_Pure_530" / "SegmentationClass"

    pick_root_img = root / "BacterialBlight_Pure_104" / "bacterialBlight_40" / "JPEGImages"
    pick_root_mask = root / "BacterialBlight_Pure_104" / "bacterialBlight_40" / "SegmentationClass"

    save_img = root / "Healthy2BacterialBlight_Pure40" / "JPEGImages"
    save_mask = root / "Healthy2BacterialBlight_Pure40" / "SegmentationClass"
    orig_list = [path.stem for path in root_img.iterdir()]
    mask_list = [path.stem for path in root_mask.iterdir()]

    pick_orig_list = [path.stem for path in pick_root_img.iterdir()]
    pick_mask_list = [path.stem for path in pick_root_mask.iterdir()]

    for name in orig_list:
        if name in mask_list:
            path_orig = root_img / (name + '.jpg')
            path_mask = root_mask / (name + '.png')
            # mask_list.remove(name)

            mask_dst_path_ = np.random.choice(pick_mask_list)  # 随机选取mask里面一张
            mask_dst_path = pick_root_mask / (mask_dst_path_ + '.png')
            img_dst_path = pick_root_img / (mask_dst_path_ + '.jpg')

            # print("选着了图片",mask_dst_path)
            img_src = cv2.imread(str(path_orig))
            mask_src = cv2.imread(str(path_mask))
            mask_dst = cv2.imread(str(mask_dst_path))
            img_dst = cv2.imread(str(img_dst_path))
            M = 1
            for i in range(1, 2):
                new_mask, new_img = new_copy_paste(mask_src, img_src, mask_dst, img_dst)
                cv2.imwrite(str(save_img / (path_orig.stem + "_" + str(i) + '.jpg')), new_img)
                cv2.imwrite(str(save_mask / (path_orig.stem + "_" + str(i) + '.png')), new_mask)
            # if str(path_orig) == str(mask_dst_path):
            #     pass
            # else:
            #     img_src = cv2.imread(str(path_orig))
            #     mask_src = cv2.imread(str(path_mask))
            #     mask_dst = cv2.imread(str(mask_dst_path))
            #     img_dst = cv2.imread(str(img_dst_path))
            # n:1(尺度抖动、翻转方向+new-copy-paste)
            # M = 1
            # for i in range(M):
            #     new_mask, new_img = new_copy_paste(mask_src, img_src, mask_dst, img_dst)
            #     cv2.imwrite(str(save_img / (path_orig.stem + "_" + str(i) + '.jpg')), new_img)
            #     cv2.imwrite(str(save_mask / (path_orig.stem + "_" + str(i) + '.png')), new_mask)


if __name__ == '__main__':
    main()

    # args = get_args()
    # main(args)

    # # 原尺度、原方向+new-copy-paste
    # mask_src = cv2.imread("data/SegmentationClass/brownspot_orig_005.png")
    # img_src = cv2.imread("data/JPEGImages/brownspot_orig_005.jpg")
    # # 需要扣出疾病的图像
    # mask_dst = cv2.imread("data/SegmentationClass/brownspot_orig_040.png")  # 输入二值图像
    # img_dst = cv2.imread("data/JPEGImages/brownspot_orig_040.jpg")
    # new_mask, new_img = new_img_add(mask_src, img_src, mask_dst, img_dst)
    # cv2.imshow("new_mask", new_mask)
    # cv2.imshow("new_img", new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 尺度抖动、翻转方向+new-copy-paste
    # mask_src = cv2.imread("data/SegmentationClass/brownspot_orig_005.png")
    # img_src = cv2.imread("data/JPEGImages/brownspot_orig_005.jpg")
    # mask_dst = cv2.imread("data/SegmentationClass/brownspot_orig_040.png")  # 输入二值图像
    # img_dst = cv2.imread("data/JPEGImages/brownspot_orig_040.jpg")
    # new_mask, new_img = new_copy_paste(mask_src, img_src, mask_dst, img_dst)
    # cv2.imshow("new_mask", new_mask)
    # cv2.imshow("new_img", new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
