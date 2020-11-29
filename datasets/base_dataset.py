from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import datasets.transforms as tr
from utils.vis import get_label_name_colors


class BaseDataset(Dataset):
    def __init__(self, img_paths, target_paths, split, base_size, crop_size, **kwargs):
        super().__init__()
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.len_dataset = len(self.img_paths)

        self.base_size = base_size  # train 基准 size
        self.crop_size = crop_size  # train, valid, test

        self.split = split
        self.transform = self.get_transform(split)

        self.num_classes = kwargs.get('num_classes', 0)
        self.bg_idx = kwargs.get('bg_idx', 0)
        self.mapbg_fn = tr.mapbg(self.bg_idx)
        self.remap_fn = tr.remap(self.bg_idx)

        if 'csv_path' in kwargs:
            self.label_names, self.label_colors = get_label_name_colors(kwargs['csv_path'])

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')

        if self.target_paths[index].endswith('.npy'):
            target = np.load(self.target_paths[index]).astype(int)
        else:
            target = np.asarray(Image.open(self.target_paths[index]), dtype=int)

        target = self.mapbg_fn(target)  # bg -> constant_bg 255
        target = Image.fromarray(target)

        sample = {
            'img': img,
            'target': target
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.img_paths)

    def get_transform(self, split):  # 交给子类完成
        pass

    def make_dataset_multiple_of_batchsize(self, batch_size):  # multiple: n. 倍数
        # 使图片数恰好为 batch_size 整数倍，不用 drop_last
        remainder = self.len_dataset % batch_size
        if remainder > 0:
            num_new_entries = batch_size - remainder
            self.img_paths.extend(self.img_paths[:num_new_entries])
            self.target_paths.extend(self.target_paths[:num_new_entries])

    def reset_dataset(self):
        self.img_paths = self.img_paths[:self.len_dataset]
        self.target_paths = self.target_paths[:self.len_dataset]
