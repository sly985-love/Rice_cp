import os, shutil  # 导入模块

# 标签文件重命名
# def copy_files():  # 定义函数名称
#     for foldName, subfolders, filenames in os.walk(path):  # 用os.walk方法取得path路径下的文件夹路径，子文件夹名，所有文件名
#         for filename in filenames:  # 遍历列表下的所有文件名 (1,5)(5,10)
#             # print(filename)
#             if str("pseudo") in filename:
#                 new_name=filename.split("_p")[0]+".png"
#                 shutil.copyfile(os.path.join(foldName, filename), os.path.join(path2, new_name))
#                 # print(filename)
#                 # print(new_name)
#
#
# if __name__ == '__main__':
#     path = r'D:\草莓项目\数据集\mildew\label'  # 运行程序前，记得修改主文件夹路径！
#     path2 = r'D:\草莓项目\数据集\mildew\label2'  # 存放文件的新路径，不要放在原路径下，不然会多复制两份
#     copy_files()  # 调用定义的函数，注意名称与定义的函数名一致


# # 将图片和标签对应
# def check_files():  # 定义函数名称
#     for foldName1, subfolders1, filenames1 in os.walk(path):
#         for foldName2, subfolders2, filenames2 in os.walk(path2):
#             for filename2 in filenames2:
#                 filename2_=filename2.split(".")[0]+".png"
#                 if filename2_ not in filenames1:
#                     print(filename2)
#                     os.remove(os.path.join(foldName2, filename2))
#
#
# if __name__ == '__main__':
#     path = r'D:\草莓项目\数据集\angular\label2'
#     path2 = r'D:\草莓项目\数据集\angular\image'
#     check_files()


# 将图片和标签对应
def check_files():  # 定义函数名称
    for foldName1, subfolders1, filenames1 in os.walk(path):
        for foldName2, subfolders2, filenames2 in os.walk(path2):
            for filename2 in filenames2:
                if filename2 in filenames1:
                    print(filename2)
                    shutil.copyfile(os.path.join(foldName2, filename2), os.path.join(path3, filename2))
                    # os.remove(os.path.join(foldName2, filename2))


if __name__ == '__main__':
    path = r'D:\草莓项目\数据集\data\VOCdevkit\VOC2012\SegmentationClass'
    path2 = r'D:\草莓项目\数据集\mildew\label'
    path3 = r'D:\草莓项目\数据集\data\VOCdevkit\VOC2012\SegmentationClass2'
    check_files()
