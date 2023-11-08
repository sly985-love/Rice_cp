import json
from pathlib import Path

json_dir = 'json_dir'  # json文件路径
out_dir = Path('out_dir1')  # 输出的 txt 文件路径


def get_json(json_file, out_dir):
    # 读取 json 文件数据
    with open(json_file, 'r') as load_f:
        content = json.load(load_f)

    image_list = content['images']
    anno_list = content['annotations']
    errsum = 0
    # 每张图片id
    for i in image_list:
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
                new_box = str(new_box)[1:-1].replace(', ', ' ')
                c_id=j["category_id"]
                str_tmp = str(c_id)+ " "+ new_box + '\n'

                tmps = tmps + str_tmp

        txt_name = i['file_name'].split('.')[0] + '.txt'
        txt_file = out_dir / txt_name
        with open(txt_file, 'w') as t:
            t.write(tmps)


get_json(r'json_dir/coco_info.json', out_dir)
