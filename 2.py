import os
import json
from pathlib import Path

json_dir = 'json_dir'  # json文件路径
out_dir = Path('out_dir1')  # 输出的 txt 文件路径
with open(r'json_dir/coco_info.json', 'r') as load_f:
    content = json.load(load_f)

image_list = content['images']
anno_list = content['annotations']
for i in image_list:
    # print(i['file_name'])
    image_id = i['id']
    width = i['width']
    height = i['height']
    # tmps = '# ' + i['file_name'] + '\n'
    tmps = ''
    # 根据图片id查注释
    # for j in anno_list:
    #     if j['image_id'] == image_id:
    #         print(image_id, j['keypoints'])


def get_json(json_file, out_dir):
    # 读取 json 文件数据
    with open(json_file, 'r') as load_f:
        content = json.load(load_f)

    image_list = content['images']
    anno_list = content['annotations']
    errsum = 0
    # 每张图片id
    for i in image_list:
        # print(i['file_name'])
        image_id = i['id']
        width = i['width']
        height = i['height']
        # tmps = '# ' + i['file_name'] + '\n'
        tmps = ''

        # 根据图片id查注释
        for j in anno_list:
            if j['image_id'] == image_id:
                # print(image_id, j['id'])
                new_box = [0, 0, 0, 0]
                box = j['bbox']
                box[0] = max(0, box[0])
                box[1] = max(0, box[1])
                box[2] = min(width - 1, box[2])
                box[3] = min(height - 1, box[3])
                new_box[0] = (box[0] + box[2] / 2) / width
                new_box[1] = (box[1] + box[3] / 2) / height
                new_box[2] = box[2] / width  # w
                new_box[3] = box[3] / height  # h
                tmp_box = new_box
                new_box = str(new_box)[1:-1].replace(', ', ' ')
                c_id=j["category_id"]
                # print(bbox)
                # keypoints = str(j['keypoints'])[1:-1].replace(', ', ' ')
                # if 'keypoints' in j:
                #     keypoints = j['keypoints']
                #     new_keys = []
                #     # print(keypoints)
                #     # 换标记
                #     if j['num_keypoints'] == 6:
                #         for k in range(0, len(keypoints)):
                #             if k % 3 == 0:
                #                 # keypoints[k] = keypoints[k] / width
                #                 new_keys.append((keypoints[k] / width))
                #             elif k % 3 == 1:
                #                 # keypoints[k] = keypoints[k] / height
                #                 new_keys.append((keypoints[k] / height))
                #             elif k % 3 == 2:
                #                 # print(keypoints[k])
                #                 if keypoints[k] == 2:
                #                     keypoints[k] = 0.0
                #     else:
                #         for k in range(0, 18):
                #             # new_keys.append('-1')
                #             if k % 3 == 0:
                #                 # keypoints[k] = keypoints[k] / width
                #                 new_keys.append((-1 / width))
                #             elif k % 3 == 1:
                #                 # keypoints[k] = keypoints[k] / height
                #                 new_keys.append((-1 / height))
                #
                #     for j, key in enumerate(new_keys):
                #         if key > 0:
                #             if j % 2 == 0:
                #                 if (tmp_box[0] + tmp_box[2] / 2) > key > (tmp_box[0] - tmp_box[2] / 2):
                #                     pass
                #                 else:
                #                     print('x:{}, {}'.format(key, tmp_box))
                #                     errsum += 1
                #                     print('这河里嘛' + i['file_name'])
                #             elif j % 2 == 1:
                #                 if (tmp_box[1] - tmp_box[3] / 2) < key < (tmp_box[1] + tmp_box[3] / 2):
                #                     pass
                #                 else:
                #                     print('y:{}, {}'.format(key, tmp_box))
                #                     errsum += 1
                #                     print('这河里嘛2' + i['file_name'])
                #
                #     new_keys = str(new_keys)[1:-1].replace(', ', ' ')
                #
                # else:
                #     new_keys = ''
                # # 拼接
                # str_tmp = '0 ' + new_box + ' ' + new_keys + '\n'
                # # print(str_tmp)
                # tmps = tmps + str_tmp

                # 拼接
                str_tmp = str(c_id)+ " "+ new_box + '\n'
                # print(str_tmp)
                tmps = tmps + str_tmp
        print(errsum)
        # print(tmps)
        # print(i['file_name'])
        txt_name = i['file_name'].split('.')[0] + '.txt'
        txt_file = out_dir / txt_name
        with open(txt_file, 'w') as t:
            t.write(tmps)


# 遍历文件夹
#
get_json(r'json_dir/coco_info.json', out_dir)
# label_path = Path(r'D:\yesy\TEA\tea_data\label')
# save_path = Path(r'D:\yesy\TEA\tea_data\label.txt')
# label_list = [path for path in label_path.iterdir()]
# for label_iter in label_list:
#     img_tmp = '# ' + str(label_iter.stem + '.jpg')
#     with open(save_path, 'r') as l:
#
#         def get_json(json_file, filename):
#             # 读取 json 文件数据
#             with open(json_file, 'r') as load_f:
#                 content = json.load(load_f)
#
#             image_list = content['images']
#
#             # # 循环处理
#             tmp = filename
#             filename_txt = out_dir + tmp + '.txt'
#             # 创建txt文件
#             fp = open(filename_txt, mode="w", encoding="utf-8")
#             # 将数据写入文件
#             str_tmp = ""  # 存储字符串内容
#
#             # 1.存文件名
#             imgname = filename + ".jpg"
#             str_tmp = "#" + " " + str(imgname) + "\n"
#             obj_num = len(content["annotations"])
#             print(content['categories'])
#             for i in range(obj_num):
#                 # 2.存目标框
#                 bbox = (content["annotations"][i])["bbox"]
#                 x = bbox[0] / content[]
#                 y = bbox[1]
#                 w = bbox[2]
#                 h = bbox[3]
#                 num_keypoints = (content["annotations"][i])["num_keypoints"]
#                 # # 3.存关键点
#                 # if num_keypoints != 6:
#                 #     x1 = -1.0
#                 #     y1 = -1.0
#                 #     s1 = -1.0
#                 #     x2 = -1.0
#                 #     y2 = -1.0
#                 #     s2 = -1.0
#                 #     x3 = -1.0
#                 #     y3 = -1.0
#                 #     s3 = -1.0
#                 #     x4 = -1.0
#                 #     y4 = -1.0
#                 #     s4 = -1.0
#                 #     x5 = -1.0
#                 #     y5 = -1.0
#                 #     s5 = -1.0
#                 #     x6 = -1.0
#                 #     y6 = -1.0
#                 #     s6 = -1.0
#                 #     # x1, y1, s1, x2, y2, s2, x3, y3, s3, x4, y4, s4, x5, y5, s5, x6, y6, s6 = -1.0
#                 # else:
#                 #     keypoints = (content["annotations"][i])["keypoints"]
#                 #     x1 = keypoints[0]
#                 #     y1 = keypoints[1]
#                 #     s1 = 0.0
#                 #     x2 = keypoints[3]
#                 #     y2 = keypoints[4]
#                 #     s2 = 0.0
#                 #     x3 = keypoints[6]
#                 #     y3 = keypoints[7]
#                 #     s3 = 0.0
#                 #     x4 = keypoints[9]
#                 #     y4 = keypoints[10]
#                 #     s4 = 0.0
#                 #     x5 = keypoints[12]
#                 #     y5 = keypoints[13]
#                 #     s5 = 0.0
#                 #     x6 = keypoints[15]
#                 #     y6 = keypoints[16]
#                 #     s6 = 0.0
#                 # # print(x1, y1, s1, x2, y2, s2, x3, y3, s3, x4, y4, s4, x5, y5, s5, x6, y6, s6)
#                 # # str(x1)+str(y1)+str(s1)+str(x2)+str(y2)+str(s2)+str(x3)+str(y3)+str(s3)+str(x4)+str(y4)+str(s4)+str(x5)+str(y5)+str(s5)+str(x6)+str(y6)+str(s6)
#                 #
#                 # str_tmp += str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + str(x1) + " " + str(
#                 #     y1) + " " + str(
#                 #     s1) + " " + str(x2) + " " + str(y2) + " " + str(s2) + " " + str(x3) + " " + str(y3) + " " + str(
#                 #     s3) + " " + str(x4) + " " + str(y4) + " " + str(s4) + " " + str(x5) + " " + str(y5) + " " + str(
#                 #     s5) + " " + str(x6) + " " + str(y6) + " " + str(s6) + "\n"
#
#             fp = open(filename_txt, mode="r+", encoding="utf-8")
#             file_str = str_tmp
#             line_data = fp.readlines()
#             if len(line_data) != 0:
#                 fp.write('\n' + file_str)
#             else:
#                 fp.write(file_str)
#             fp.close()
#
#
# files = os.listdir(json_dir)  # 得到文件夹下的所有文件名称
# s = []
# for file in files:  # 遍历文件夹
#     filename = file.split('.')[0]
#     get_json(json_dir + "/" + file, filename)
