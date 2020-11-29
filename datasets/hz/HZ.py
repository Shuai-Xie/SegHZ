import os
from utils.misc import read_txt_as_list
from datasets.base_dataset import BaseDataset
from datasets.transforms import get_transform, get_hzcolor_transform
from PIL import Image
from datasets.hz.helper import get_merge_func
import numpy as np

this_dir = os.path.dirname(__file__)


class HZ_Merge(BaseDataset):

    def __init__(self, root, split, base_size, crop_size, merge_all_buildings=False):
        img_paths = read_txt_as_list(os.path.join(root, f'{split}_img_paths.txt'))
        target_paths = read_txt_as_list(os.path.join(root, f'{split}_target_paths.txt'))

        num_classes = 7 if merge_all_buildings else 10
        super().__init__(
            img_paths, target_paths, split, base_size, crop_size,
            num_classes=num_classes,
            bg_idx=0,
            csv_path=os.path.join(this_dir, f'hz{num_classes}.csv')
        )
        self.merge_fn = get_merge_func(merge_all_buildings)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')

        if self.target_paths[index].endswith('.npy'):
            target = np.load(self.target_paths[index]).astype(int)
        else:
            target = np.asarray(Image.open(self.target_paths[index]), dtype=int)

        target = self.merge_fn(target)
        target = self.mapbg_fn(target)
        target = Image.fromarray(target.astype('uint8'))
        target = target.resize(img.size, Image.NEAREST)  # 统一 target 和 img 尺寸; 256 -> 1024 ?

        sample = {
            'img': img,
            'target': target,
            'path': self.img_paths[index]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_transform(self, split):
        return get_transform(split, self.base_size, self.crop_size)


class HZ20(BaseDataset):
    def __init__(self, root, split, base_size, crop_size):
        img_paths = read_txt_as_list(os.path.join(root, f'{split}_img_paths.txt'))
        target_paths = read_txt_as_list(os.path.join(root, f'{split}_target_paths.txt'))

        super().__init__(
            img_paths, target_paths, split, base_size, crop_size,
            num_classes=20,
            bg_idx=0,
            csv_path=os.path.join(this_dir, 'hz20.csv')
        )

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')

        if self.target_paths[index].endswith('.npy'):
            target = np.load(self.target_paths[index]).astype(int)
        else:
            target = np.asarray(Image.open(self.target_paths[index]), dtype=int)

        target = self.mapbg_fn(target)
        target = Image.fromarray(target.astype('uint8'))
        target = target.resize(img.size, Image.NEAREST)  # 统一 target 和 img 尺寸

        sample = {
            'img': img,
            'target': target,
            'path': self.img_paths[index]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_transform(self, split):
        return get_transform(split, self.base_size, self.crop_size)


class HZ_ColorLabel(BaseDataset):

    def __init__(self, root, split, base_size, crop_size, merge_all_buildings=False):
        img_paths = read_txt_as_list(os.path.join(root, f'{split}_img_paths.txt'))
        target_paths = read_txt_as_list(os.path.join(root, f'{split}_target_paths.txt'))

        num_classes = 7 if merge_all_buildings else 10
        super().__init__(
            img_paths, target_paths, split, base_size, crop_size,
            num_classes=num_classes,
            bg_idx=0,
            csv_path=os.path.join(this_dir, f'hz{num_classes}.csv')
        )
        self.merge_fn = get_merge_func(merge_all_buildings)

    def __getitem__(self, index):
        if self.target_paths[index].endswith('.npy'):
            target = np.load(self.target_paths[index]).astype(int)
        else:
            target = np.asarray(Image.open(self.target_paths[index]), dtype=int)

        target = self.merge_fn(target)  # 7 cls, bg=0
        img = self.label_colors[target]  # 160,160
        img = Image.fromarray(img)
        img = img.resize((self.crop_size, self.crop_size))  # 512,512

        target = self.mapbg_fn(target)
        target = Image.fromarray(target)
        target = target.resize(img.size, Image.NEAREST)  # 统一 target 和 img 尺寸

        sample = {
            'img': img,
            'target': target,
            'path': self.img_paths[index]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_transform(self, split):
        return get_hzcolor_transform(split, self.base_size, self.crop_size)
