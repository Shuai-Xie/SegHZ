import numpy as np
from utils.misc import read_txt_as_list, dump_pickle, load_pickle
import os
import matplotlib.pyplot as plt

"""
jianggan18, crop: 14 x 14 = 196
river18, crop: 14 x 14 = 196
xiaoshan18, crop: 14 x 17 = 238
xihu18, crop: 14 x 17 = 238
airport18, crop: 14 x 17 = 238

airport18_9_4.png, 按照 _ split 即可
"""


def write_data_split():
    data_dist = {
        'jianggan18': np.zeros((14, 14)),
        'river18': np.zeros((14, 14)),
        'xiaoshan18': np.zeros((14, 17)),
        'xihu18': np.zeros((14, 17)),
        'airport18': np.zeros((14, 17)),
    }

    train_imgs = read_txt_as_list('/datasets/RS_Dataset/HZ20/z18_512/train_img_paths.txt')  # 0
    valid_imgs = read_txt_as_list('/datasets/RS_Dataset/HZ20/z18_512/valid_img_paths.txt')  # 1
    test_imgs = read_txt_as_list('/datasets/RS_Dataset/HZ20/z18_512/test_img_paths.txt')  # 2

    for p in valid_imgs:
        img, i, j = os.path.basename(p).replace('.png', '').split('_')
        data_dist[img][int(i)][int(j)] = 1

    for p in test_imgs:
        img, i, j = os.path.basename(p).replace('.png', '').split('_')
        data_dist[img][int(i)][int(j)] = 2

    dump_pickle(data_dist, 'data_dist.pkl')


def plt_data_dist():
    data_dist = load_pickle('data_dist.pkl')
    f, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i, (img, dist) in enumerate(sorted(data_dist.items(), key=lambda t: t[0])):
        axs.flat[i].imshow(dist, cmap='tab10')
        axs.flat[i].axis('off')
        # axs.flat[i].set_title(img)

    # f.tight_layout()  # 调整整体空白
    plt.show()


def plt_data_accs():
    data_acc = load_pickle('runs/HZ/bise_sp256_ffm256_deconv_Aug09_100655/data_acc.pkl')
    f, axs = plt.subplots(1, 5, figsize=(15, 3))
    for idx, (img, patch_accs) in enumerate(sorted(data_acc.items(), key=lambda t: t[0])):
        patch_accs[0][0][1] = 0.3
        patch_accs[-1][-1][1] = 1.0
        axs.flat[idx].imshow(patch_accs[:, :, 1], cmap='jet')
        axs.flat[idx].axis('off')
        for i in range(patch_accs.shape[0]):
            for j in range(patch_accs.shape[1]):
                axs.flat[idx].text(j, i, str(int(patch_accs[i][j][0])),
                                   horizontalalignment="center",
                                   verticalalignment='center',
                                   color='black' if patch_accs[i][j][1] > 0.5 else 'white',
                                   fontsize=7)
        axs.flat[idx].set_title('%.2f' % (np.mean(patch_accs[:, :, 1])))

        # if idx == 4:
        #     f.colorbar(axmap, ax=axs.flat[idx])

    # f.tight_layout()  # 调整整体空白
    # plt.subplots_adjust(wspace=0)
    plt.show()


def sat_acc():
    data_acc = load_pickle('runs/HZ/bise_sp256_ffm256_deconv_Aug09_100655/data_acc.pkl')
    for img, patch_accs in data_acc.items():
        print(img, np.mean(patch_accs[:, :, 1]))


if __name__ == '__main__':
    # write_data_split()
    # plt_data_dist()
    # plt_data_accs()
    sat_acc()
    pass
