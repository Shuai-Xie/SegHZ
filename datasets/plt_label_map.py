import numpy as np
import matplotlib.pyplot as plt
from utils.vis import get_label_name_colors


def plt_label_map(label_names, label_colors, rows, cols, row_height, col_width, square_len=20,
                  figsize=(10, 8), xytext=(+18, -11), fig_title=None):
    """
    :param label_names: lable name list
    :param label_colors: label color list
    :param rows: num of figure rows
    :param cols: num of figure cols
    :param row_height: height of each row
    :param col_width: width of each col
    :param figsize: overall figure size, like (10, 8)
    :param fig_title: figure title, like ADE20K-150class
    :return:
    """

    # create label_color map
    label_map = np.ones((row_height * rows, col_width * cols, 3), dtype='uint8') * 255
    cnt = 0
    for i in range(rows):  # 1st row is black = background
        for j in range(cols):
            if cnt >= len(label_colors):  # in case, num of lables < rows * cols
                break
            beg_pix = (i * row_height, j * col_width)
            end_pix = (beg_pix[0] + square_len, beg_pix[1] + square_len)  # 20 is color square side
            label_map[beg_pix[0]:end_pix[0], beg_pix[1]:end_pix[1]] = label_colors[cnt]  # RGB
            cnt += 1

    plt.figure(figsize=figsize, dpi=100)
    plt.imshow(label_map)

    # add label_name
    cnt = 0
    for i in range(rows):  # 1st row is black = background
        for j in range(cols):
            if cnt >= len(label_names):  # in case, num of lables < rows * cols
                break
            beg_pix = (j * col_width, i * row_height)  # note! (y,x)
            plt.annotate(f'{cnt + 1}_{label_names[cnt]}',
                         xy=beg_pix, xycoords='data', xytext=xytext, textcoords='offset points',
                         color='k')
            cnt += 1

    plt.axis('off')
    if fig_title:
        plt.title(fig_title + '\n', fontweight='black')  # 上移一段距离，哈哈

    # plt.savefig('{}.png'.format(fig_title), bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    # # VOC20
    # label_names, label_colors = get_label_name_colors(csv_path='csv/voc20.csv')
    # plt_label_map(label_names, label_colors, rows=2, cols=10, row_height=30, col_width=200, figsize=(16, 3),
    #               fig_title='CamVid-32class')

    # CamVid
    # label_names, label_colors = get_label_name_colors(csv_path='camvid/camvid11.csv')
    # plt_label_map(label_names, label_colors, rows=2, cols=6, row_height=30, col_width=200, figsize=(8, 3),
    #               fig_title='CamVid-11class')

    # Cityscapes
    # label_names, label_colors = get_label_name_colors(csv_path='cityscapes/cityscapes19.csv')
    # plt_label_map(label_names, label_colors, rows=2, cols=10, row_height=30, col_width=200, figsize=(16, 3),
    #               fig_title='Cityscapes-19class')

    # DeepGlobe LandCover
    # label_names, label_colors = get_label_name_colors(csv_path='deepglobe/land6.csv')
    # plt_label_map(label_names, label_colors, rows=1, cols=7,
    #               square_len=40,
    #               row_height=50, col_width=240, figsize=(8, 1))

    # hz buildings
    label_names, label_colors = get_label_name_colors(csv_path='datasets/hz/hz7.csv')
    plt_label_map(label_names[1:], label_colors[1:],
                  rows=7, cols=1,
                  square_len=20, xytext=(+22, -12),
                  row_height=30, col_width=400, figsize=(6, 6))

    # # SUN-RGBD
    # label_names, label_colors = get_label_name_colors(csv_path='sunrgbd/sunrgbd37.csv')
    # plt_label_map(label_names, label_colors, rows=4, cols=10, row_height=30, col_width=200, figsize=(14, 3),
    #               fig_title='SUNRGBD-37class')

    # # ADE20K
    # label_names, label_colors = get_label_name_colors(csv_path='csv/ade150.csv')
    # plt_label_map(label_names, label_colors, rows=10, cols=15, row_height=30, col_width=200, figsize=(22, 4),
    #               fig_title='ADE20K-150class')
