import numpy as np
from pathlib import Path
from PIL import Image

# （3）将3通道24位的彩色mask转换成单通道8位的灰度图
# 每个像素值值代表每一类，后面在训练的时候网络也是根据这个像素值分类的
label = {'background': 0, 'leaf': 1, 'angular': 2, 'mildew': 3} # 这儿里没写完，背景+叶片+5类病害=7，所以标签应该是0-6，对应的elif也需要增加
path_root = Path(r'D:\SegmentationClass')  # 3通道24位的彩色标签的存放路径，注意这里是覆盖，所以做好备份
path_list = [path for path in path_root.iterdir()]
for jpath in path_list:
    if jpath.suffix == '.png':
        img = Image.open(jpath)
        img = img.convert('L')
        w, h = img.size
        img = np.array(img)
        for j in range(w):
            for i in range(h):
                if img[i][j] == 0:
                    continue
                elif img[i][j] == 100:  # 你可以print(img[i][j] == 0)看看你的叶子和病害的像素值分别是多少，（这个100是我的标签叶子是100，我们的绿不一样，你需要自己打印看看）
                    img[i][j] = label['leaf']
                elif img[i][j] == 51:
                    img[i][j] = label['angular']
                elif img[i][j] == 65:
                    img[i][j] = label['mildew']
        img_pil = Image.fromarray(img, mode='L')
        img_pil.save(jpath)
        print(jpath)
