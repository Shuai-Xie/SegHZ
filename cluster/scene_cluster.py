from cluster.encoder import MobileEncoder
from datasets.build_datasets import build_hz_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from models import HRNet
from demo.base import *
from utils.misc import *
import cv2

device = torch.device('cuda:1')

"""
编码数据 image -> vector
"""


def encode_dataset():
    # encoder
    encoder = MobileEncoder(output_stride=16).to(device).eval()
    out_dim = 320

    # dataset 小尺寸 256 编码特征
    trainset, testset = build_hz_datasets(base_size=256, crop_size=256)  # 1106/140
    trainset.transform = testset.transform  # normalize + totensor

    @torch.no_grad()
    def encode(dataset):
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        res = torch.zeros((0, out_dim)).to(device)

        for sample in tqdm(dataloader):
            image = sample['img'].to(device)
            vecs = encoder(image)
            res = torch.cat((res, vecs), dim=0)

        return res.cpu().numpy()

    np.save('cluster/train_features.npy', encode(trainset))  # (1106,320)
    np.save('cluster/test_features.npy', encode(testset))  # (140,320)


def dataset_error_score():
    from utils.misc import recover_color_img
    from utils.vis import plt_img_target_pred_error
    from datasets.build_datasets import data_cfg

    label_names, label_colors = data_cfg['HZ_Merge']['label_colors']
    label_names, label_colors = label_names[1:], label_colors[1:]  # 去掉 bg

    model = HRNet('model/hrnet/cfg/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml', use_pretrain=False)
    load_state_dict(model, f'runs/HZ_Merge/hrnet_dce512_Aug26_215738/checkpoint.pth.tar', device)

    # dataset
    trainset, testset = build_hz_datasets(base_size=512, crop_size=512)
    trainset.transform = testset.transform

    @torch.no_grad()
    def error_score(dataset):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        res = []

        for sample in tqdm(dataloader):
            if 'xihu18_12' not in sample['path'][0]:
                continue

            img, target = sample['img'], sample['target']
            img = img.to(device)

            pred = predict_whole(model, img, use_flip=True, return_pred=True)
            pred = to_numpy(pred, toint=True)

            target = target.squeeze(0).numpy()  # h,w
            pred = cv2.resize(pred, target.shape[::-1], interpolation=cv2.INTER_NEAREST)

            # error
            error_mask = np.zeros_like(target)
            mask = pred != target
            error_mask[mask] = target[mask]

            error = np.sum(mask) / target.size

            print(error, sample['path'][0])
            ori_img = recover_color_img(img.squeeze(0))
            plt_img_target_pred_error(ori_img, target, pred, error_mask,
                                      label_colors,
                                      title='{} {}'.format(os.path.basename(sample['path'][0]), error))
            res.append(error)

        exit(0)

        return np.array(res)

    np.save('cluster/train_errors.npy', error_score(trainset))
    np.save('cluster/test_errors.npy', error_score(testset))


"""
tSNE/PCA 降维直观去看数据分布
ListedColormap([[0, 0, 1.], [1.0, 0, 0]], name='br')
"""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# 降维
def tsne_reduce_dim():
    train_features = np.load('cluster/train_features.npy')
    test_features = np.load('cluster/test_features.npy')

    data = np.vstack((train_features, test_features))  # (N,C)
    # embedded to low-dim space, n_components 指定维度
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(data)  # 返回2D空间坐标 (N,2)
    np.save('cluster/X_tsne.npy', X_tsne)


# 可视分布
def vis_data_dist():  # N,C
    X_tsne = np.load('cluster/X_tsne.npy')

    # train/test error scores
    train_errors = np.load('cluster/train_errors.npy')
    test_errors = np.load('cluster/test_errors.npy')
    target_error = np.hstack((train_errors, test_errors))

    # train/test tag 0/1
    target_split = np.array([0] * len(train_errors) + [1] * len(test_errors))

    plt.subplots(1, 2, figsize=(10, 5))
    s = 20

    plt.subplot(1, 2, 1)
    mask = target_split == 0
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c='b', s=s, label='train')
    mask = ~ mask  # np 数组取反
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c='r', s=s, label='test')
    plt.title('data')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target_error, s=s, cmap='jet')
    plt.title('error')

    # x < -30
    x, y, w, h = -50, -30, 20, 30
    plt.gca().add_patch(plt.Rectangle(xy=(x, y),
                                      width=w, height=h,
                                      edgecolor='k',
                                      fill=False, linewidth=2))

    plt.subplots_adjust(right=0.93)  # 调节前面已画子图占比
    plt.colorbar(cax=plt.axes([0.95, 0.12, 0.02, 0.76]))
    plt.clim(0, 1)

    plt.show()


def extract_bad_case():
    from utils.misc import read_txt_as_list, mkdir
    import shutil

    X_tsne = np.load('cluster/X_tsne.npy')
    bad_idxs = [idx for idx, t in enumerate(X_tsne) if t[0] < -40 and t[1] < -20]
    print(bad_idxs)

    # train/test error scores
    train_errors = np.load('cluster/train_errors.npy')
    test_errors = np.load('cluster/test_errors.npy')
    target_error = np.hstack((train_errors, test_errors))

    print(target_error[bad_idxs])

    # read img and copy
    root = '/datasets/RS_Dataset/HZ20'
    # bad_dir = os.path.join(root, 'bad_case')
    # mkdir(bad_dir)

    # 1106 + 140 = 1246
    img_paths = read_txt_as_list(os.path.join(root, 'train_img_paths.txt')) + \
                read_txt_as_list(os.path.join(root, 'test_img_paths.txt'))
    img_paths = [img_paths[i] for i in bad_idxs]
    for p in img_paths:
        print(p)
    exit(0)

    for idx in bad_idxs:
        print(f'{root}/{img_paths[idx]}')
        # shutil.copy(src=f'{root}/{img_paths[idx]}', dst=bad_dir)  # copy 支持目录为 dir, copyfile 必须是路径文件名

    pass


if __name__ == '__main__':
    # encode_dataset()
    dataset_error_score()
    # tsne_reduce_dim()
    # vis_data_dist()
    # extract_bad_case()
    pass
