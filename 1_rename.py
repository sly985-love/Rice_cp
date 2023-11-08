import os, shutil


# （1）将彩色标签文件重命名：去掉文件名中的pseudo，使标签名与图片名字对应
def copy_files():
    for foldName, subfolders, filenames in os.walk(path):
        for filename in filenames:
            if str("pseudo") in filename:
                new_name = filename.split("_p")[0] + ".png"
                shutil.copyfile(os.path.join(foldName, filename), os.path.join(path2, new_name))


if __name__ == '__main__':
    path = r'D:\草莓项目\数据集\mildew\label'  # 存放彩色标签的文件夹路径
    path2 = r'D:\草莓项目\数据集\mildew\label2'  # 存放重命名后的彩色标签文件夹路径
    copy_files()
