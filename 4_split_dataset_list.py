# 数据集划分
# 每类病害以8：2的比例划分为train和val,并存txt在ImageSet下
import glob
import os.path
import argparse
import warnings
import numpy as np


# （4）数据集划：分别对每类病害进行数据划分（默认train:val=8:2）
# step1:先将每类病害以一下文件格式存储
#    eg：mildew（某类病害）
#              JPEGImages（里面存放该类病害的图片）
#              SegmentationClass（里面放该类病害的灰度图标签）
# step2：修改路径①②③
# step3：会在mildew文件目录下生成一个train.txt,val.txt，test.txt
# step4：自己新建一个trainval.txt，手动复制train.txt,val.txt里面的内容

# step5：对每类病害完成step1-step4工作
# step6：创建以下文件夹：data\VOCdevkit\VOC2012\ImageSets（这个data文件就是最后模型训练需要的所有数据）
#                   ImageSets下创建：JPEGImages（里面存放所有病害的所有图片）
#                   ImageSets下创建：SegmentationClass（里面存放所有病害的所有标签）
#                   ImageSets下创建：Segmentation（里面存放train.txt，val.txt，trainval.txt）
# （注意！！！！这里的train.txt，val.txt，trainval.txt，是从每类病害（step3-4）那里复制合成的新的，复制txt的时候一定要注意，最后一行没有非空行或者空格，有就删掉）

def parse_args():
    parser = argparse.ArgumentParser(
        description='A tool for proportionally randomizing dataset to produce file lists.'
    )
    parser.add_argument('--dataset_root', help='the dataset root path', type=str,
                        default=r'D:\草莓项目\数据集\mildew\split')  # ①
    parser.add_argument(
        '--images_dir_name', help='the directory name of images', type=str,
        default=r'D:\草莓项目\数据集\mildew\split\JPEGImages')  # ②
    parser.add_argument(
        '--labels_dir_name', help='the directory name of labels', type=str,
        default=r'D:\草莓项目\数据集\mildew\split\SegmentationClass')  # ③
    parser.add_argument(
        '--split', help='', nargs=3, type=float, default=[0.8, 0.2, 0])
    parser.add_argument(
        '--label_class',
        help='label class names',
        type=str,
        nargs='*',
        default=['__background__', '__foreground__'])
    parser.add_argument(
        '--separator',
        dest='separator',
        help='file list separator',
        default=" ",
        type=str)
    parser.add_argument(
        '--format',
        help='data format of images and labels, e.g. jpg, tif or png.',
        type=str,
        nargs=2,
        default=['jpg', 'png'])
    parser.add_argument(
        '--postfix',
        help='postfix of images or labels',
        type=str,
        nargs=2,
        default=['', ''])

    return parser.parse_args()


def get_files(path, format, postfix):
    pattern = '*%s.%s' % (postfix, format)

    search_files = os.path.join(path, pattern)
    search_files2 = os.path.join(path, "*", pattern)  # 包含子目录
    search_files3 = os.path.join(path, "*", "*", pattern)  # 包含三级目录

    filenames = glob.glob(search_files)
    filenames2 = glob.glob(search_files2)
    filenames3 = glob.glob(search_files3)

    filenames = filenames + filenames2 + filenames3

    return sorted(filenames)


def generate_list(args):
    separator = args.separator
    dataset_root = args.dataset_root
    if sum(args.split) != 1.0:
        raise ValueError("划分比例之和必须为1")

    file_list = os.path.join(dataset_root, 'labels.txt')
    with open(file_list, "w") as f:
        for label_class in args.label_class:
            f.write(label_class + '\n')

    image_dir = os.path.join(dataset_root, args.images_dir_name)
    label_dir = os.path.join(dataset_root, args.labels_dir_name)
    image_files = get_files(image_dir, args.format[0], args.postfix[0])
    label_files = get_files(label_dir, args.format[1], args.postfix[1])
    if not image_files:
        warnings.warn("No files in {}".format(image_dir))
    num_images = len(image_files)

    if not label_files:
        warnings.warn("No files in {}".format(label_dir))
    num_label = len(label_files)

    if num_images != num_label and num_label > 0:
        raise Exception("Number of images = {}    number of labels = {} \n"
                        "Either number of images is equal to number of labels, "
                        "or number of labels is equal to 0.\n"
                        "Please check your dataset!".format(num_images,
                                                            num_label))

    image_files = np.array(image_files)
    label_files = np.array(label_files)
    state = np.random.get_state()
    np.random.shuffle(image_files)
    np.random.set_state(state)
    np.random.shuffle(label_files)

    start = 0
    num_split = len(args.split)
    dataset_name = ['train', 'val', 'test']
    for i in range(num_split):
        dataset_split = dataset_name[i]
        print("Creating {}.txt...".format(dataset_split))
        if args.split[i] > 1.0 or args.split[i] < 0:
            raise ValueError("{} dataset percentage should be 0~1.".format(
                dataset_split))

        file_list = os.path.join(dataset_root, dataset_split + '.txt')
        with open(file_list, "w") as f:
            num = round(args.split[i] * num_images)
            end = start + num
            if i == num_split - 1:
                end = num_images
            for item in range(start, end):
                left = image_files[item].replace(dataset_root, '')
                if left[0] == os.path.sep:
                    left = left.lstrip(os.path.sep)
                    # 只要文件夹名
                    left_new = left.split("\\")[1]
                    left_new = left_new.split(".")[0]
                    print(left_new)
                    # try:
                    #     right = label_files[item].replace(dataset_root, '')
                    #     if right[0] == os.path.sep:
                    #         right = right.lstrip(os.path.sep)
                    #     line = left + separator + right + '\n'
                    # except:
                    line = left_new + '\n'

                f.write(line)
                # print(line)
            start = end


if __name__ == '__main__':
    args = parse_args()
    generate_list(args)
