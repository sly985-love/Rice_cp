import cv2

try:
    from PIL import Image
except ImportError:
    import Image
import os
from PIL import Image
import numpy as np
import cv2 as cv
from sklearn.svm import SVC


def extract_disease(img):
    img_arr = np.asarray(img, np.float64)
    # 选取水稻叶片与背景图像上的关键点RGB值(10个)
    bg_RGB = np.array(
        [[211, 215, 232], [118, 105, 61, ], [173, 176, 181], [125, 188, 37, ], [97, 88, 57],
         [100, 162, 35], [36, 51, 10], [159, 200, 42], [167, 167, 177], [99, 165, 75],
         [189, 188, 183], [190, 184, 178], [190, 171, 172], [194, 185, 214], [173, 158, 137]]
    )
    # 选取水稻病害的关键点RGB值(10个)
    tea_RGB = np.array(
        [[255, 237, 144], [233, 218, 117], [190, 172, 130], [249, 208, 128], [213, 193, 132],
         [201, 201, 111], [208, 104, 127], [237, 190, 139], [255, 234, 155], [255, 227, 171],
         [224, 229, 170], [253, 211, 139], [221, 167, 97], [209, 196, 119], [245, 213, 172]]
    )
    RGB_arr = np.concatenate((bg_RGB, tea_RGB), axis=0)
    label = np.append(np.zeros(bg_RGB.shape[0]), np.ones(tea_RGB.shape[0]))
    img_reshape = img_arr.reshape([img_arr.shape[0] * img_arr.shape[1], img_arr.shape[2]])
    svc = SVC(kernel='poly', degree=7)
    svc.fit(RGB_arr, label)
    predict = svc.predict(img_reshape)
    bg_bool = predict == 0.
    bg_bool = bg_bool[:, np.newaxis]
    bg_bool_3col = np.concatenate((bg_bool, bg_bool, bg_bool), axis=1)
    bg_bool_3d = bg_bool_3col.reshape((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
    img_arr[bg_bool_3d] = 0.
    img_split = Image.fromarray(img_arr.astype('uint8'))
    img_split.save('split0.jpg')
    tea_s = cv.imread('split0.jpg')
    bgr = cv.cvtColor(tea_s, cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    median = cv.medianBlur(thresh, 3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(median, cv.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

    return closing


def extract_leaf(img):
    img_arr = np.asarray(img, np.float64)
    bg_RGB = np.array(
        [[181, 186, 189], [186, 183, 204, ], [192, 183, 202], [172, 171, 167, ], [169, 161, 158],
         [149, 141, 130], [204, 195, 214], [157, 151, 156], [139, 136, 103], [65, 62, 27],
         [42, 39, 48], [44, 35, 58], [53, 39, 53], [53, 33, 50], [41, 18, 49],
         [44, 42, 54], [43, 35, 66], [36, 22, 73], [52, 38, 53], [55, 36, 55]]
    )
    tea_RGB = np.array(
        [[138, 188, 55], [48, 86, 27], [102, 173, 71], [176, 193, 148], [154, 185, 46],
         [239, 212, 123], [161, 176, 13], [212, 175, 133], [79, 113, 62], [149, 206, 49]]
    )
    RGB_arr = np.concatenate((bg_RGB, tea_RGB), axis=0)
    label = np.append(np.zeros(bg_RGB.shape[0]), np.ones(tea_RGB.shape[0]))
    img_reshape = img_arr.reshape([img_arr.shape[0] * img_arr.shape[1], img_arr.shape[2]])
    svc = SVC(kernel='poly', degree=7)
    svc.fit(RGB_arr, label)
    predict = svc.predict(img_reshape)
    bg_bool = predict == 0.
    bg_bool = bg_bool[:, np.newaxis]
    bg_bool_3col = np.concatenate((bg_bool, bg_bool, bg_bool), axis=1)
    bg_bool_3d = bg_bool_3col.reshape((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
    img_arr[bg_bool_3d] = 0.
    img_split = Image.fromarray(img_arr.astype('uint8'))
    img_split.save('split1.jpg')
    tea_s = cv.imread('split1.jpg')
    bgr = cv.cvtColor(tea_s, cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    median = cv.medianBlur(thresh, 3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(median, cv.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

    return closing


def add_disease_leaf_mask(img, disease_mask, leaf_mask):
    H = disease_mask.shape[0]
    W = disease_mask.shape[1]
    mask31 = np.zeros_like(img)
    mask31[:, :, 0] = disease_mask
    mask31[:, :, 1] = disease_mask
    mask31[:, :, 2] = disease_mask
    mask32 = np.zeros_like(img)
    mask32[:, :, 0] = leaf_mask
    mask32[:, :, 1] = leaf_mask
    mask32[:, :, 2] = leaf_mask
    mask = mask31
    center = [0, 0]
    for i in range(H):
        for j in range(W):
            # print(mask31[i, j])
            # 红色：rgb(255, 0, 0)
            # 绿色：rgb(0, 255, 0)
            if mask31[i, j][0] == 255:
                mask31[i, j][0] = 255
                mask31[i, j][1] = 0
                mask31[i, j][2] = 0
            if mask32[i, j][0] == 255:
                mask32[i, j][0] = 0
                mask32[i, j][1] = 255
                mask32[i, j][2] = 0
            if mask[i, j][0] == 255:
                mask32[center[0] + i, center[1] + j] = mask31[i, j]
    return mask32


def get_mask(img):
    disease_mask = extract_disease(img)
    leaf_mask = extract_leaf(img)
    mask = add_disease_leaf_mask(img, disease_mask, leaf_mask)
    return mask


if __name__ == '__main__':
    img_path = 'images/src/'
    save_img_path = 'images/save/'
    for img_name in os.listdir(img_path):
        # print(img_path+img_name)
        img = Image.open(img_path + img_name)
        mask = get_mask(img)
        cv2.imwrite(save_img_path + img_name, mask)

