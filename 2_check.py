import os


# （2）检查标签和图片是否对应：对照标签删除没有标签的图片文件
def check_files():
    for foldName1, subfolders1, filenames1 in os.walk(path):
        for foldName2, subfolders2, filenames2 in os.walk(path2):
            for filename2 in filenames2:
                filename2_ = filename2.split(".")[0] + ".png"
                if filename2_ not in filenames1:
                    print(filename2)
                    os.remove(os.path.join(foldName2, filename2))


if __name__ == '__main__':
    path = r'D:\草莓项目\数据集\angular\label2'  # 存放标签文件的文件目录
    path2 = r'D:\草莓项目\数据集\angular\image'  # 存放图片文件的文件目录
    check_files()
