from utils import read_txt_as_list, get_label_name_colors, plt_img_target
from datasets.bdci2017.helper import data_mapper
import cv2

label_names, label_colors = get_label_name_colors('datasets/hz/hz7.csv')
img_paths = [
    '/datasets/RS_Dataset/BDCI/BDCI2017-jiage/train/1',  # 7939*7969
    '/datasets/RS_Dataset/BDCI/BDCI2017-jiage/train/2',  # 7939*7969
    '/datasets/RS_Dataset/BDCI/BDCI2017-jiage-Semi/train/2',  # 2470*4011
    '/datasets/RS_Dataset/BDCI/BDCI2017-jiage-Semi/train/3',  # 6116*3357
]


def map_data(cls):
    for k, v in data_mapper.items():
        cls[cls == k] = v
    return cls


def vis_data():
    for p in img_paths:
        img_path = p + '_visible.png'
        cls_path = p + '_class.png'

        img = cv2.imread(img_path)
        cls = cv2.imread(cls_path, cv2.IMREAD_ANYDEPTH)
        # img = cv2.resize(img, cls.shape)
        cls = cv2.resize(cls, img.shape[:2][::-1], cv2.INTER_NEAREST)
        cls = map_data(cls)

        plt_img_target(img, cls, label_colors)


# todo: 在此小数据集上训练效果
def crop_data():
    for p in img_paths:
        img_path = p + '_visible.png'
        cls_path = p + '_class.png'

        img = cv2.imread(img_path)
        cls = cv2.imread(cls_path, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, cls.shape[::-1])
        cls = map_data(cls)
        plt_img_target(img, cls, label_colors)


crop_data()
