import os
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def save_img(image, path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i += 1

    image.save(path_pattern % i)


def aug_img12(orig_arr, mask_arr):
    transform = A.Compose(
        [
            # 对比度亮度拉伸
            A.RandomBrightnessContrast(p=1),

        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]
def aug_img3(orig_arr, mask_arr):
    transform = A.Compose(
        [
            # 锐化
            A.Sharpen(p=1),
        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]
def aug_img4(orig_arr, mask_arr):
    transform = A.Compose(
        [
            # 椒盐噪声
            A.MultiplicativeNoise(p=1),
        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]
def aug_img5(orig_arr, mask_arr):
    transform = A.Compose(
        [
            A.GlassBlur(p=1),
        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]
def aug_img6(orig_arr, mask_arr):
    transform = A.Compose(
        [
            A.Blur(p=1),
        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]
def aug_img7(orig_arr, mask_arr):
    transform = A.Compose(
        [
            A.HorizontalFlip(p=1),
        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]
def aug_img8(orig_arr, mask_arr):
    transform = A.Compose(
        [
            A.VerticalFlip(p=1),
        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]
def aug_img9(orig_arr, mask_arr):
    transform = A.Compose(
        [

            A.Rotate(p=1),
        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]
def aug_img10(orig_arr, mask_arr):
    transform = A.Compose(
        [
            A.RandomScale(p=1),
        ]
    )
    augmentations = transform(image=orig_arr, mask=mask_arr)
    return augmentations["image"], augmentations["mask"]

def main():
    root = Path('data')
    root_img = root / "JPEGImages"
    root_mask = root / "SegmentationClass"
    save_img_mask = root / "save"
    times = 10

    orig_list = [path.stem for path in root_img.iterdir()]
    mask_list = [path.stem for path in root_mask.iterdir()]

    for name in orig_list:
        if name in mask_list:
            path_orig = root_img / (name + '.jpg')
            path_mask = root_mask / (name + '.png')

            orig = Image.open(path_orig)
            mask = Image.open(path_mask)

            mask_palette = mask.convert("P").getpalette()

            arr_orig = np.array(orig)
            arr_mask = np.array(mask)
# 数据增强
            aug_orig, aug_mask = aug_img12(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))

            aug_orig, aug_mask = aug_img3(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))

            aug_orig, aug_mask = aug_img4(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))

            aug_orig, aug_mask = aug_img5(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))

            aug_orig, aug_mask = aug_img6(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))

            aug_orig, aug_mask = aug_img7(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))

            aug_orig, aug_mask = aug_img8(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))

            aug_orig, aug_mask = aug_img9(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))

            aug_orig, aug_mask = aug_img10(arr_orig, arr_mask)
            img_orig = Image.fromarray(aug_orig)
            img_mask = Image.fromarray(aug_mask)
            img_mask.putpalette(mask_palette)
            save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
            save_img(img_mask, str(save_img_mask / (path_mask.stem + '_%s.png')))
            # for i in range(times):
            #     aug_orig, aug_mask = aug_img(arr_orig, arr_mask)
            #
            #     img_orig = Image.fromarray(aug_orig)
            #     img_mask = Image.fromarray(aug_mask)
            #
            #     img_mask.putpalette(mask_palette)
            #
            #     save_img(img_orig, str(root_img / (path_orig.stem + '_%s.jpg')))
            #     save_img(img_mask, str(root_mask / (path_mask.stem + '_%s.png')))
                # save_img(img_orig, str(save_img_mask / (path_orig.stem + '_%s.jpg')))
                # save_img(img_mask, str(root_mask / (path_mask.stem + '_%s.png')))



if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
# import numpy as np
