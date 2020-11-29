import numpy as np
from glob import glob
import os
import cv2
from utils.misc import write_list_to_txt, mkdir
import math


class DatasetMaker:
    def __init__(self, root, z):
        """
        :param root: 存放 不同 z 的 .png 和 .npy
        :param z: 等级
        """
        self.root = root
        self.z = z

    def make_dataset(self, crop_size, overlap):
        """
        :param crop_size: crop patch image size
        :param overlap: overlaping ~ (0, 1)
        """
        save_img_dir = f'{self.root}/image'
        save_msk_dir = f'{self.root}/mask'
        mkdir(save_img_dir)
        mkdir(save_msk_dir)

        for npy_path in glob(f'{self.root}/*.npy'):  # npy 存放只有 1 份
            base_name = f"{os.path.basename(npy_path).replace('.npy', '')}{self.z}"  # sat z img
            img = cv2.imread(f'{self.root}/{base_name}.png')  # RGB 图片，对应不同的 z
            target = np.load(npy_path).astype(int)
            self.slide_crop_img(img, target, crop_size, overlap,
                                save_img_dir, save_msk_dir, base_name)  # crop 保存的 basename 保留 z 等级

        # save image/msk paths
        img_paths, target_paths = [], []
        for img in os.listdir(save_img_dir):
            if img == '@eaDir':
                continue
            img_paths.append(f'{save_img_dir}/{img}')
            target_paths.append(f"{save_msk_dir}/{img.replace('.jpeg', '.npy')}")

        write_list_to_txt(img_paths, f'{self.root}/img_paths.txt')
        write_list_to_txt(target_paths, f'{self.root}/target_paths.txt')

    @staticmethod
    def slide_crop_img(img, target, crop_size, overlap, save_img_dir, save_msk_dir, basename=''):
        """
        :param img:     h,w,c
        :param target:  h,w
        :param crop_size: size of cropped small img
        :param overlap: 交叠比，越高，切分出图片越多
        :param save_img_dir: 保存路径
        :param save_msk_dir:
        :param basename: 图片基本名，如 airport18
        :return:
        """
        H, W, _ = img.shape  # 按照瓦片图拼接尺寸，将 target 与 img 对齐
        if target.shape != (H, W):  # todo: 考虑别的 上采样方式, 减弱最近邻的锯齿
            target = cv2.resize(target, (W, H), interpolation=cv2.INTER_NEAREST)  # 不改变标签值 resize

        stride = int(math.ceil(crop_size * (1 - overlap)))  # overlap -> stride
        tile_rows = int(math.ceil((H - crop_size) / stride) + 1)  # 向上取整 ceil，全部区域
        tile_cols = int(math.ceil((W - crop_size) / stride) + 1)
        num_tiles = tile_rows * tile_cols
        print("%s, need %i x %i = %i prediction tiles @ stride %i px" % (basename, tile_cols, tile_rows, num_tiles, stride))

        for row in range(tile_rows):
            for col in range(tile_cols):
                x2, y2 = min(col * stride + crop_size, W), min(row * stride + crop_size, H)
                x1, y1 = max(int(x2 - crop_size), 0), max(int(y2 - crop_size), 0)

                crop_img = img[y1:y2, x1:x2, :]
                crop_target = target[y1:y2, x1:x2]

                img_name = f'{basename}_{row}_{col}'
                cv2.imwrite(f'{save_img_dir}/{img_name}.jpeg', crop_img)
                np.save(f'{save_msk_dir}/{img_name}.npy', crop_target)

                print('\r==> {}/{}'.format(row * tile_cols + col + 1, num_tiles), end='')
        print()


def mk_dataset():
    a = DatasetMaker(root='/datasets/RS_Dataset/HZ20/train/scenes', z=18)
    a.make_dataset(crop_size=1024, overlap=0)


if __name__ == '__main__':
    mk_dataset()
    pass
