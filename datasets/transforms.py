import torch
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import torch.nn.functional as F
import constants


def mapbg(bg_idx):
    """
    image bg 转成 constants.BG_INDEX, 类别从 [0,..,C-1]
    """

    # bg 在首部，需要调整 实际类别 前移1位
    def map_headbg(target):
        target = target.astype(int)
        target -= 1  # 1->0
        target[target == -1] = constants.BG_INDEX  # 255
        return target.astype('uint8')

    # bg 在尾部，直接替换为 constant 即可
    def map_other(target):
        target = target.astype(int)
        target[target == bg_idx] = constants.BG_INDEX
        return target.astype('uint8')

    if bg_idx == 0:
        return map_headbg
    else:
        return map_other


def remap(bg_idx):
    """
    分割结果 -> 回归原始 bg idx，方面 vis
    """

    def remap_headbg(target):
        target = target.astype(int)
        target += 1
        target[target == constants.BG_INDEX + 1] = bg_idx
        return target.astype('uint8')

    def remap_other(target):
        target = target.astype(int)
        target[target == constants.BG_INDEX] = bg_idx
        return target.astype('uint8')

    if bg_idx == 0:
        return remap_headbg
    else:
        return remap_other


class Compose:  # 可以采用 默认的
    def __init__(self, trans_list):
        self.trans_list = trans_list

    def __call__(self, sample):
        for t in self.trans_list:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('

        for t in self.trans_list:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class RandomHorizontalFlip:
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 0.5:
            img = img.transpose(0)
            target = target.transpose(0)

        sample['img'], sample['target'] = img, target
        return sample


class RandomVerticalFlip:
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 0.5:
            img = img.transpose(1)
            target = target.transpose(1)

        sample['img'], sample['target'] = img, target
        return sample


class RandomRightAngle:
    """随机旋转直角, 90/180/270"""

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 0.5:
            k = random.randint(2, 4)
            img = img.transpose(k)
            target = target.transpose(k)

        sample['img'], sample['target'] = img, target
        return sample


class RandomDiagnoal:
    """随机对角线转换，主/副"""

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 10:
            k = random.randint(5, 6)  # 闭区间
            img = img.transpose(k)
            target = target.transpose(k)

        sample['img'], sample['target'] = img, target
        return sample


class RandomRotate:
    def __init__(self, degree):  # 旋角上限
        self.degree = degree

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        target = target.rotate(rotate_degree, Image.NEAREST, fillcolor=constants.BG_INDEX)

        sample['img'], sample['target'] = img, target
        return sample


class RandomGaussianBlur:
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random() * self.radius))  # radius ~ [0,1]

        sample['img'], sample['target'] = img, target
        return sample


class GaussianBlur:
    def __init__(self, radius=0.5):
        self.radius = radius

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        sample['img'], sample['target'] = img, target
        return sample


class RandomScaleCrop:
    def __init__(self, base_size, crop_size, scales=(0.8, 1.2)):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        # 原图 scale
        short_size = random.randint(int(self.base_size * self.scales[0]), int(self.base_size * self.scales[1]))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # random scale
        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)

        # scale 后短边 < 要 crop 尺寸，补图
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  # img fill 0, 后面还有 normalize
            target = ImageOps.expand(target, border=(0, 0, padw, padh), fill=constants.BG_INDEX)  # target fill bg_idx

        # random crop
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)

        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        sample['img'], sample['target'] = img, target
        return sample


class RandomBuildingFGCrop:
    """
    避免 crop 出 包含 bg 太多的 patch
    random base_num = grid_size **2
    """

    def __init__(self, base_size, crop_size, over_lap=0.5,
                 downsample=1,
                 fg_thre=0.1, dataset='HZ'):
        self.base_size = base_size
        self.crop_size = crop_size  # patch size
        self.stride = int(self.crop_size * (1 - over_lap))  # 64
        self.grid_size = 1 + (base_size - crop_size) // self.stride
        self.fg_thre = fg_thre
        self.dataset = dataset
        self.downsample = downsample

        self.hz20_no_buildings = [1, 2, 3, 4, 5, 6, 7, 8, 19, 20]
        self.hz_merge_no_buildings = [1, 2, 3, 4, 5, 6, 11]  # 7,8,9,10 building

    @staticmethod
    def _binary_by_buildings(target, no_buildings):
        for idx in no_buildings:
            target[target == idx] = 0  # bg
        target[target != 0] = 1  # fg_idx 置1，计算 fg_rate
        return target

    def _rand_fg_coord(self, target):
        tmp = np.array(target)

        if self.dataset == 'HZ':  # 如果 hz20 数据，需要 map 其他类到 0
            tmp += 1  # remap bg
            tmp = self._binary_by_buildings(tmp, self.hz20_no_buildings)
            tmp = torch.from_numpy(tmp).unsqueeze(0).unsqueeze(0).float()  # 为了用 avgpool
        elif self.dataset == 'HZ_Merge':
            tmp += 1
            tmp = self._binary_by_buildings(tmp, self.hz_merge_no_buildings)
            tmp = torch.from_numpy(tmp).unsqueeze(0).unsqueeze(0).float()
        elif self.dataset == 'HZ_Building':
            tmp[tmp != 0] = 1

        fg_rates = F.avg_pool2d(tmp,
                                kernel_size=self.crop_size // self.downsample,
                                stride=self.stride // self.downsample).reshape(-1).numpy()  # 1D fg_rate

        top_idxs = np.argsort(fg_rates)[::-1]
        keep_num = np.sum(fg_rates >= self.fg_thre)
        # choose = random.choice(top_idxs[:keep_num]) if keep_num > 0 else top_idxs[0]  # HZ_Building
        choose = random.choice(top_idxs[:keep_num]) if keep_num > 0 else top_idxs[0]  # HZ v1
        # choose = random.choice(top_idxs[:keep_num]) if keep_num > 0 else random.choice(top_idxs)  # HZ v2
        return choose, fg_rates[choose]

    def _rand_shift_coord(self, x, y):
        bound = int(self.stride * 0.5)  # 向右 向下 即能覆盖基本所有区域
        rand_x = random.randint(0, bound)
        rand_y = random.randint(0, bound)

        # left-top
        if x + rand_x < self.base_size - self.crop_size:
            x += rand_x
        if y + rand_y < self.base_size - self.crop_size:
            y += rand_y
        return x, y

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        chose_idx, fg_rate = self._rand_fg_coord(target)

        # 从 chose_idx 还原 crop patch 坐标
        i, j = chose_idx // self.grid_size, chose_idx % self.grid_size
        x1, y1 = self.stride * j, self.stride * i

        # rand shift for crop patch
        # x1, y1 = self._rand_shift_coord(x1, y1)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1 // self.downsample, y1 // self.downsample,
                              (x1 + self.crop_size) // self.downsample,
                              (y1 + self.crop_size) // self.downsample))
        # target = np.array(target)
        # if self.dataset == 'HZ_Building' and fg_rate < self.fg_thre:  # 处理 HZ_Building
        #     target[:] = constants.BG_INDEX  # 全部忽略，这部分样本 loss 忽略
        # target = Image.fromarray(target)

        sample['img'], sample['target'] = img, target
        return sample


class FixScaleCrop:
    def __init__(self, crop_size):  # valid, 固定原图 aspect，crop 图像中央
        self.crop_size = min(crop_size)

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)

        w, h = img.size  # 放缩后的 size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        sample['img'], sample['target'] = img, target
        return sample


class FixedResize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        img = img.resize(self.size, Image.BILINEAR)
        target = target.resize(self.size, Image.NEAREST)

        sample['img'], sample['target'] = img, target
        return sample


class ColorJitter:
    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])

        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)

        sample['img'], sample['target'] = img, target
        return sample


class Normalize:
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        img = np.array(img).astype(np.float32)
        target = np.array(target).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        sample['img'], sample['target'] = img, target
        return sample


class ToTensor:
    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        target = np.array(target).astype(np.float32)

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).long()

        sample['img'], sample['target'] = img, target
        return sample


def get_transform(split, base_size, crop_size=None):
    if split == 'train':
        return Compose([
            # sampler
            RandomScaleCrop(base_size, crop_size, scales=(0.8, 1.0)),  # 增加小尺度推理能力
            # flip
            RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            RandomRightAngle(),
            # RandomDiagnoal(),
            # color
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            RandomGaussianBlur(),
            # normal
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
    elif split == 'valid':
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
    elif split == 'test':
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


def get_hzcolor_transform(split, base_size, crop_size=None):
    if split == 'train':
        return Compose([
            # sampler
            # RandomScaleCrop(base_size, crop_size, scales=(0.8, 1.0)),  # 增加小尺度推理能力
            # flip
            RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomRightAngle(),
            # RandomDiagnoal(),
            # color
            # ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            # RandomGaussianBlur(),
            # normal
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
    elif split == 'valid':
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
    elif split == 'test':
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


import torchvision.transforms as transforms


def get_img_transfrom(base_size=None):
    trans = [transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    if base_size is not None:
        trans.insert(0, transforms.Resize(base_size))
    return transforms.Compose(trans)
