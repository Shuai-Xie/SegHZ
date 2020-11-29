import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
from tqdm import tqdm
import shutil
from utils.misc import mkdir

hz_features = np.load('cluster/hz_features.npy')  # (3840, 320)
X_tsne = np.load('cluster/X_tsne.npy')


# tSNE/PCA 降维直观去看数据分布
def reduce_dim():
    # embedded to low-dim space, n_components 指定维度
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(hz_features)  # 降到2D
    X_pca = PCA(n_components=2).fit_transform(hz_features)
    np.save('cluster/X_tsne.npy', X_tsne)
    np.save('cluster/X_pca.npy', X_pca)


# 2D 查看数据分布
def vis_dist(n_clusters):
    target = np.load(f'cluster/hz_labels_c{n_clusters}.npy')

    plt.figure(figsize=(6, 6))
    # cmap = plt.cm.get_cmap('rainbow', n_clusters)
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target, s=10, cmap=cmap)  # c: 颜色序列
    plt.title(f'clusters: {n_clusters}')
    cbar = plt.colorbar()
    cbar.set_ticks(range(n_clusters))
    plt.clim(-0.5, n_clusters - 0.5)
    plt.show()


# 聚类
def cluster_img(n_clusters=5, max_iter=300):  # 300/1000 结果一致
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=0)
    model.fit(hz_features)

    target = model.labels_
    np.save(f'cluster/hz_labels_c{n_clusters}.npy', target)
    # print(model.inertia_)  # 到簇心欧式距离和，簇越多越小

    clsnum = np.bincount(target, minlength=n_clusters)
    return clsnum.var()  # 各类别数量是否均衡


def demo_clusters():
    # cls_vars = []
    # for c in tqdm(range(2, 21)):
    #     cvar = cluster_img(n_clusters=c)
    #     cls_vars.append(cvar)
    #     vis_dist(n_clusters=c)
    # cls_vars = np.array(cls_vars)
    # np.save(f'cluster/cls_vars.npy', cls_vars)

    # cls_vars = np.load(f'cluster/cls_vars.npy')
    # plt.plot(range(5, 21), cls_vars[4:])
    # plt.xlabel('n_clusters')
    # plt.xticks(range(5, 21))
    # plt.ylabel('var_cls_num')
    # plt.show()

    vis_dist(n_clusters=10)
    # vis_dist(n_clusters=11)
    # vis_dist(n_clusters=17)
    # vis_dist(n_clusters=20)


def separate_scenes(n_clusters=7):
    target = np.load(f'cluster/hz_labels_c{n_clusters}.npy')

    root = '/nfs2/sontal/RS_images/16'
    img_paths = [os.path.join(root, p) for p in sorted(os.listdir(root))]

    base_dir = '/datasets/RS_Dataset/HZ20'
    for idx in range(n_clusters):
        mkdir(f'{base_dir}/scenes/class_{idx}')

    for idx, src in tqdm(enumerate(img_paths)):
        shutil.copy(src, dst=f'{base_dir}/scenes/class_{target[idx]}')


def grid_scenes(n_clusters=10):
    target = np.load(f'cluster/hz_labels_c{n_clusters}.npy')
    target = target.reshape((60, 64)).T  # 先列展开
    target = np.flipud(target)

    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(target, cmap=plt.cm.get_cmap('tab20', n_clusters))
    plt.show()


if __name__ == '__main__':
    # reduce_dim()
    # demo_clusters()
    # vis_dist(n_clusters=7)
    # separate_scenes(n_clusters=10)
    grid_scenes(n_clusters=10)
    # vis_dist(n_clusters=5)
    pass
