import numpy as np
import numba as nb
from datasets.transforms import mapbg
from datasets.hz.helper import get_merge_func
from constants import hz_dir
import os
from utils import get_label_name_colors, read_txt_as_list, entropy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

num_classes = 7


def save_img_stats():
    merge_fn = get_merge_func(merge_all_buildings=True)
    mapbg_fn = mapbg(bg_idx=0)

    target_paths = read_txt_as_list(os.path.join(hz_dir, 'train_target_paths.txt'))

    img_freqs = []
    for path in tqdm(target_paths):
        y = np.load(path)
        y = merge_fn(y)
        y = mapbg_fn(y)  # 0-6, 255

        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        freq = count_l / count_l.sum()
        img_freqs.append(freq)

    img_freqs = np.array(img_freqs)
    np.save('datasets/hz/img_cls_freqs.npy', img_freqs)


@nb.jit(nopython=True)
def greedy_entropy(img_freqs):
    def entropy(p):
        return np.nansum(-p * np.log(p))

    print(img_freqs.shape)  # (1000, 17)

    group_num = img_freqs.shape[0]
    use = [False] * group_num

    group_orders = []
    group_entros = []

    base_freqs = np.zeros((num_classes,))

    for i in range(group_num):
        print(i)
        max_freqs = base_freqs
        max_entro = 0.  # 只找 当前情况下最大，不用指定 先验最大 entropy
        max_idx = -1

        pre_freqs = base_freqs * i

        for j in range(group_num):
            if use[j]:
                continue
            # *i 表示已经确定的 i 个 group 的整体 freq, 保证和 j 同样量纲
            tmp_freqs = pre_freqs + img_freqs[j]
            tmp_freqs = tmp_freqs / tmp_freqs.sum()  # 注意归一化！
            tmp_entro = entropy(tmp_freqs)
            if tmp_entro >= max_entro:
                max_freqs = tmp_freqs
                max_entro = tmp_entro
                max_idx = j

        # 使用 max 更新 base 情况
        base_freqs = max_freqs
        group_entros.append(max_entro)  # 当前数量下的 entropy
        group_orders.append(max_idx)
        use[max_idx] = True

    return group_orders, group_entros


def cal_greedy_entropy():
    # cal greedy_entropy with numba
    img_freqs = np.load('datasets/hz/img_cls_freqs.npy')

    t1 = time.time()
    group_orders, group_entros = greedy_entropy(img_freqs)
    t2 = time.time()
    print('exe:', t2 - t1)  # 1.8885393142700195s

    np.save('datasets/hz/img_orders.npy', group_orders)
    np.save('datasets/hz/img_entros.npy', group_entros)


def plt_ordered_entropy():
    group_entros = np.load('datasets/hz/img_entros.npy')
    plt.plot(range(1, 1 + len(group_entros)), group_entros, color='k')
    plt.xlabel('img_num')
    plt.ylabel('dataset_class_entropy')

    train_num = 1505
    xy = (train_num, group_entros[train_num - 1])
    plt.plot(xy[0], xy[1], 'ro')  # red s: square, o: circle
    plt.annotate('trainset: %.3f' % float(xy[1]), xy=xy,
                 xycoords='data', xytext=(-100, 0),
                 textcoords='offset points', fontsize=12)

    train_num = 300
    xy = (train_num, group_entros[train_num - 1])
    plt.plot(xy[0], xy[1], 'o', color='green')  # red s: square, o: circle
    plt.annotate('validset: %.3f' % float(xy[1]), xy=xy,
                 xycoords='data', xytext=(+20, 0),
                 textcoords='offset points', fontsize=12)

    plt.show()


if __name__ == '__main__':
    # save_img_stats()
    # cal_greedy_entropy()
    plt_ordered_entropy()
