from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import get_label_name_colors, read_txt_as_list, entropy
import os
from datasets.transforms import mapbg
from datasets.hz.helper import get_merge_func
from constants import hz_dir

label_names, label_colors = get_label_name_colors('datasets/hz/hz7.csv')
num_classes = 7


def plt_freq_bar(frequency, title=''):
    print(frequency)
    entro = entropy(frequency)
    print('entropy:', entro)

    plt.figure(figsize=(5, 4))
    x, y = range(len(frequency)), frequency

    colors = [[c / 255 for c in label_colors[i + 1]] for i in range(num_classes)]
    plt.bar(x, y, width=0.6, color=colors)

    # x 轴标签
    plt.xticks(x, range(1, num_classes + 1))
    plt.ylim([0, 0.5])
    # y 轴数字标签
    for a, b in zip(x, y):
        plt.text(a, b + 0.002, '%.3f' % b, ha='center', va='bottom', fontsize=10)

    plt.title(f'{title} - entropy: {entro}')
    plt.show()


def vis_freqs():
    from datasets.hz.config import old_788, new_1505

    plt_freq_bar(old_788['freqs'], title='Old-788')
    plt_freq_bar(new_1505['freqs'], title='New-1505')


def dataset_stats():
    merge_fn = get_merge_func(merge_all_buildings=True)
    mapbg_fn = mapbg(bg_idx=0)

    target_paths = read_txt_as_list(os.path.join(hz_dir, 'train_target_paths.txt'))
    z = np.zeros((num_classes,))

    ordered_img_idxs = np.load('datasets/hz/img_orders.npy')
    ordered_img_idxs = ordered_img_idxs[:300]

    for idx in tqdm(ordered_img_idxs):
        y = np.load(target_paths[idx])
        y = merge_fn(y)
        y = mapbg_fn(y)  # 0-6, 255

        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l

    # freqs
    total_frequency = np.sum(z)
    freqs = z / total_frequency

    plt_freq_bar(freqs, title='validset 300')


def dataset_stats_ori():
    merge_fn = get_merge_func(merge_all_buildings=True)
    mapbg_fn = mapbg(bg_idx=0)

    target_paths = read_txt_as_list(os.path.join(hz_dir, 'valid_target_paths.txt'))
    z = np.zeros((num_classes,))

    for path in tqdm(target_paths):
        y = np.load(path)
        y = merge_fn(y)
        y = mapbg_fn(y)  # 0-6, 255

        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l

    # freqs
    total_frequency = np.sum(z)
    freqs = z / total_frequency

    plt_freq_bar(freqs, title='validset 300')


if __name__ == '__main__':
    # dataset_stats()
    dataset_stats_ori()
    # vis_freqs()
    pass
