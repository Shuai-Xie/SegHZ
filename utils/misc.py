import os
import time
import numpy as np
import torch
import sys
import pickle
import json
import constants
import shutil
import math


def entropy(p):
    return np.nansum(-p * np.log(p))


def to_numpy(var, toint=False):
    #  Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    if isinstance(var, torch.Tensor):
        var = var.squeeze().detach().cpu().numpy()
    if toint:
        var = var.astype('uint8')
    return var


# pickle io
def dump_pickle(data, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
        print('write data to', out_path)


def load_pickle(in_path):
    with open(in_path, 'rb') as f:
        data = pickle.load(f)  # list
        return data


# json io
def dump_json(adict, out_path):
    with open(out_path, 'w', encoding='UTF-8') as json_file:
        # 设置缩进，格式化多行保存; ascii False 保存中文
        json_str = json.dumps(adict, indent=2, ensure_ascii=False)
        json_file.write(json_str)


def load_json(in_path):
    with open(in_path, 'rb') as f:
        adict = json.load(f)
        return adict


# io: txt <-> list
def write_list_to_txt(a_list, txt_path):
    with open(txt_path, 'w') as f:
        for p in a_list:
            f.write(p + '\n')


def read_txt_as_list(f):
    with open(f, 'r') as f:
        return [p.replace('\n', '') for p in f.readlines()]


def approx_print(arr, decimals=2):
    arr = np.around(arr, decimals)
    print(','.join(map(str, arr)))


def recover_color_img(img):
    """
    cvt tensor image to RGB [note: not BGR]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.transpose(img, axes=[1, 2, 0])  # h,w,c
    img = img * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)  # 直接通道相成?
    img = (img * 255).astype('uint8')
    return img


def colormap(N=256, normalized=False):
    """
    return color
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# dropout
def turn_on_dropout(module):
    if type(module) == torch.nn.Dropout:
        module.train()


def turn_off_dropout(module):
    if type(module) == torch.nn.Dropout:
        module.eval()


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time


def max_normalize_to1(a):
    return a / (np.max(a) + 1e-12)


def minmax_normalize(a):  # min/max -> [0,1]
    min_a, max_a = np.min(a), np.max(a)
    return (a - min_a) / (max_a - min_a)


def cvt_mask_to_score(mask, pixel_scores):  # len(pixel_scores) = num_classes
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    valid = mask != constants.BG_INDEX
    class_cnts = np.bincount(mask[valid], minlength=len(pixel_scores))  # 0-5
    diver_score = np.sum(pixel_scores * class_cnts) / class_cnts.sum()
    return diver_score


class Logger:
    """logger"""

    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='UTF-8')  # 打开时自动清空文件

    def write(self, msg):
        self.terminal.write(msg)  # 命令行打印
        self.log.write(msg)

    def flush(self):  # 必有，不然 AttributeError: 'Logger' object has no attribute 'flush'
        pass

    def close(self):
        self.log.close()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccCaches:
    """acc cache queue"""

    def __init__(self, patience):
        self.accs = []  # [(epoch, acc), ...]
        self.patience = patience

    def reset(self):
        self.accs = []

    def add(self, epoch, acc):
        if len(self.accs) >= self.patience:  # 先满足 =
            self.accs = self.accs[1:]  # 队头出队列
        self.accs.append((epoch, acc))  # 队尾添加

    def full(self):
        return len(self.accs) == self.patience

    def max_cache_acc(self):
        max_id = int(np.argmax([t[1] for t in self.accs]))  # t[1]=acc
        max_epoch, max_acc = self.accs[max_id]
        return max_epoch, max_acc


class LossThre:
    # epoch 越靠后，越不确信即 p 越大的点，越可能是噪声
    def __init__(self, min_p=0.01, max_p=0.1, epochs=100):
        self.min_thre = -math.log(max_p)  # p 越大，阈值越低，判断此项为噪声 主观性越强
        self.max_thre = -math.log(min_p)  # p 越小，阈值越高，判断此项为噪声 可能性越强
        self.gap = self.max_thre - self.min_thre
        self.epochs = epochs

    def __call__(self, epoch):  # [1, -1]
        return 0.5 * self.gap * (1 + math.cos(1.0 * epoch / self.epochs * math.pi)) + self.min_thre


def demo_loss_thre():
    x = list(range(100))
    loss_thre = LossThre()
    y = [loss_thre(e) for e in x]
    print(y[0], y[-1])
    print(loss_thre.max_thre, loss_thre.min_thre)

    import matplotlib.pyplot as plt

    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    demo_loss_thre()
    pass
