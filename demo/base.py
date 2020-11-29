import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt


def replace_module_in_state(state_dict):
    keys = list(state_dict.keys())
    for k in keys:
        if 'module' in k:
            state_dict[k[7:]] = state_dict.pop(k)
    return state_dict


def load_state_dict(model, ckpt_path, device='cpu'):
    # 默认在 cuda:0 就容易报错
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(replace_module_in_state(ckpt['state_dict']))
    print('load', ckpt_path, 'epoch', ckpt['epoch'])


@torch.no_grad()
def postprocess(model, inputs,
                lr=False, tb=False, lrtb=False, diag=False, diag2=False):
    out = model(inputs)
    cnt = 1
    if lr:
        out_lr = model(inputs.flip(dims=[3])).flip(dims=[3])
        out += out_lr
        cnt += 1
    if tb:
        out_tb = model(inputs.flip(dims=[2])).flip(dims=[2])
        out += out_tb
        cnt += 1
    if lrtb:
        out_lrtb = model(inputs.flip(dims=[2, 3])).flip(dims=[2, 3])
        out += out_lrtb
        cnt += 1
    if diag:  # 主对角线
        out_diag = model(inputs.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out += out_diag
        cnt += 1
    if diag2:
        out_diag2 = model(
            inputs.permute(0, 1, 3, 2).flip(dims=[2, 3])
        ).permute(0, 1, 3, 2).flip(dims=[2, 3])
        out += out_diag2
        cnt += 1
    return out / cnt


def predict_sliding(model, image,
                    num_classes, crop_size,
                    overlap=0.25,  # 控制 infer 数量
                    return_pred=True):
    """
    滑窗 infer 大图
    @param model:
    @param image:
    @param num_classes:
    @param crop_size: 大图 crop 小图，crop_size = model input size
    @param overlap:
    @param return_pred: reture pred if True else return probs
    @return:
    """
    B, _, H, W = image.shape
    # print('img size: {} x {}'.format(H, W))

    # out_stirde 控制模型 输出 size， 开辟存储空间，保存输出
    full_probs = torch.zeros((B, num_classes, H, W)).cuda()
    cnt_preds = torch.zeros((B, num_classes, H, W)).cuda()

    # row/col 滑窗范围
    stride = int(math.ceil(crop_size * (1 - overlap)))  # overlap -> stride
    tile_rows = int(math.ceil((H - crop_size) / stride) + 1)
    tile_cols = int(math.ceil((W - crop_size) / stride) + 1)
    num_tiles = tile_rows * tile_cols
    # print("Need %i x %i = %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, num_tiles, stride))

    for row in range(tile_rows):
        for col in range(tile_cols):
            # bottom-right / left-top 保证右下有效，反推左上
            x2, y2 = min(col * stride + crop_size, W), min(row * stride + crop_size, H)
            x1, y1 = max(int(x2 - crop_size), 0), max(int(y2 - crop_size), 0)

            # crop input img
            img = image[:, :, y1:y2, x1:x2]

            out = postprocess(model, img, lr=True)
            out = F.upsample(out, (crop_size, crop_size), mode='bilinear', align_corners=True)  # 应对 1/4 output
            # probs = F.softmax(out, dim=1)  # 使用 softmax 归一化后的 acc 更准确

            # map image pos -> output pos
            full_probs[:, :, y1:y2, x1:x2] += out  # C 维 out 之和，为了正常计算 valid loss
            cnt_preds[:, :, y1:y2, x1:x2] += 1  # 对应 pixel 估计次数

            # print('\r==> {}/{}'.format(row * tile_cols + col + 1, num_tiles), end='')

    full_probs /= cnt_preds

    if return_pred:
        preds = torch.argmax(full_probs, dim=1)
        return preds
    else:
        return full_probs


def plt_class_evals(res, label_colors, title=''):
    accs, ious = res['accs'] * 100, res['ious'] * 100

    xs = np.arange(len(accs))

    if len(accs) in [9, 11, 20]:
        plt.figure(figsize=(14 * len(accs) / 20, 4), dpi=100)
        cls_colors = label_colors[:]
        x_labels = range(1, 21)
    else:  # merge buildings
        plt.figure(figsize=(10, 4), dpi=100)
        cls_colors = label_colors[:9] + label_colors[15:]  # 9+5
        x_labels = list(range(1, 10)) + list(range(16, 21))

    width = 0.4
    fontsize = 8
    rotation = 0

    for idx, (x, y) in enumerate(zip(xs, accs)):
        plt.bar(x - 0.2, y, width=width, align='center',  # 底部 tick 对应位置
                linewidth=1, edgecolor=[0.7, 0.7, 0.7],
                color=[a / 255.0 for a in cls_colors[idx]])
        plt.text(x - 0.2, y + 0.2,
                 s='%.2f' % y,
                 rotation=rotation,
                 ha='center', va='bottom', fontsize=fontsize)

    for idx, (x, y) in enumerate(zip(xs, ious)):
        plt.bar(x + 0.2, y, width=width,
                linewidth=1, edgecolor=[0.7, 0.7, 0.7],
                color=[a / 255.0 for a in cls_colors[idx]])
        plt.text(x + 0.2, y + 0.2,
                 s='%.2f' % y,
                 rotation=rotation,
                 ha='center', va='bottom', fontsize=fontsize)

    plt.xticks(xs, x_labels, size='small')
    plt.ylim([0, 100])
    plt.title("{}. Acc: {:.2f}, mAcc: {:.2f}, mIoU: {:.2f}".format(
        title, res['Acc'] * 100, res['mAcc'] * 100, res['mIoU'] * 100))
    plt.show()


def plt_confusion_matrix(cm, label_names, normalize=True):
    import itertools
    C = cm.shape[0]

    accuracy = np.trace(cm) / float(np.sum(cm))

    cm[[1, 2], 0] = 1  # smooth 两类
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]  # C,C / C,1 广播机制
    cm[[1, 2], 0] = 0.

    plt.figure(figsize=(8, 6), dpi=200)
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

    thresh = 0.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(C), range(C)):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment='center',
                     # color='black',
                     fontsize=7,
                     color="white" if cm[i, j] > thresh else "black"
                     )
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color='black',
                     fontsize=7,
                     # color="white" if cm[i, j] > thresh else "black"
                     )

    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

    if C == 20 or C == 11:
        x_labels = range(1, C + 1)
        y_labels = label_names
    else:
        x_labels = list(range(1, 10)) + list(range(16, 21))
        y_labels = label_names[:9] + label_names[15:]

    plt.xticks(range(C), x_labels, fontsize=7)
    plt.yticks(range(C), y_labels, fontsize=7)

    plt.ylim(-0.5, C - 0.5)
    plt.gca().invert_yaxis()  # 要在 ticks 之后，反转 y lim
    plt.show()
