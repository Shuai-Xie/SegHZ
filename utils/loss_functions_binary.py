"""
binary image segmentation
"""
import torch
import torch.nn.functional as F

epsilon = 1e-5


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W) 将 C 通道带出
    """
    if len(tensor.size()) == 1:  # N
        return tensor
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)  # C,N


"""Distribution-based"""


def bce(y_true, y_pred):
    return F.binary_cross_entropy(y_pred, y_true)


# 给正类 loss 加权，相对的
# loss = FP + FN 两项
# pos_weight > 1, FN 惩罚相对弱, 会出现更多 FN，即 pred 区域更大
def weighted_bce(y_true, y_pred, alpha=0.25):  # alpha: 正样本占比
    pos_weight = alpha / (1 - alpha)
    loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pos_mask = y_true == 1.0
    loss[pos_mask] = loss[pos_mask] * pos_weight
    return loss.mean()
    # 只用于 logits, 即 sigmoid 之前的结果; 内部会连带计算 sigmoid 同时求导; 如果直接传入 prob，会得到错误结果
    # loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=pos_weight, reduction='none')
    # print(loss)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):  # alpha: 正样本占比, gamma: 困难系数
    pos_mask = y_true == 1.0
    neg_mask = ~pos_mask

    bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce_loss)
    loss = (1 - pt) ** gamma * bce_loss  # 统一 正负样本 形式的 focal loss

    if not pos_mask.any():  # 无正样本
        return loss.mean() * (1 - alpha)
    elif not neg_mask.any():
        return loss.mean() * alpha
    else:
        return alpha * loss[pos_mask].mean() + (1 - alpha) * loss[neg_mask].mean()


def focal_loss2(y_true, y_pred, alpha=0.25, gamma=2):
    pos_mask = y_true == 1.0
    neg_mask = ~pos_mask

    pos_loss, neg_loss = 0., 0.
    if pos_mask.any():
        y_pos = y_pred[pos_mask]
        pos_loss = - ((1 - y_pos) ** gamma * torch.log(y_pos)).mean()
    if neg_mask.any():
        y_neg = y_pred[neg_mask]
        neg_loss = - (y_neg ** gamma * torch.log(1 - y_neg)).mean()

    return alpha * pos_loss + (1 - alpha) * neg_loss


"""Region-based"""


# 敏感度: 正类被正确估计的占比 = recall
# TP / TP + FN
def sensitivity(y_true, y_pred):
    TP = (y_true * y_pred).sum()  # zero out soft TP
    GT_P = y_true.sum()
    return TP / (GT_P + epsilon)


# 特异度：负类正确估计的占比
# TN / TP + FN
def specificity(y_true, y_pred):
    TN = ((1 - y_true) * (1 - y_pred)).sum()
    GT_N = (1 - y_true).sum()
    return TN / (GT_N + epsilon)


def confusion(y_true, y_pred):
    TP = (y_true * y_pred).sum()  # 取出 soft TP
    FP = ((1 - y_true) * y_pred).sum()  # GT_N * Pred_P = FP
    FN = (y_true * (1 - y_pred)).sum()  # GT_P * Pred_N = FN
    TN = ((1 - y_true) * (1 - y_pred)).sum()

    precision = TP / (TP + FP + epsilon)  # 准确率
    recall = TP / (TP + FN + epsilon)

    sensitivity = recall
    specificity = TN / (TN + FP + epsilon)

    return precision, recall, sensitivity, specificity


# IoU = A ∩ B / A ∪ B
def iou(y_true, y_pred):  # B,C,H,W
    inter = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - inter  # 真正并集
    return inter / (union + epsilon)


def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)


# Dice = 2 * A ∩ B / |A| + |B|
def dice(y_true, y_pred):  # B,C,H,W
    inter = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()  # 交集部分 + 2次，所以分 母交集 *2
    # union = (y_true ** 2).sum() + (y_pred ** 2).sum()
    return 2 * inter / (union + epsilon)


def dice_loss(y_ture, y_pred):
    return 1 - dice(y_ture, y_pred)


# 将 dice loss 拓展到多分类，并用 w 使 loss 更加稳定
def general_dice(y_true, y_pred):  # B,C,H,W
    y_true = flatten(y_true)  # C,N
    y_pred = flatten(y_pred)

    w = y_true.sum(-1)  # 计算各类正例数量
    w = 1 / (w ** 2 + epsilon)  # C

    numerator = (y_true * y_pred).sum(-1)  # C
    denominator = (y_true + y_pred).sum(-1)

    # C 维度 加权和 之比; 所以 w 没有消掉
    return 2 * (w * numerator).sum() / ((w * denominator).sum() + epsilon)


def general_dice_loss(y_true, y_pred):
    return 1 - general_dice(y_true, y_pred)


# 将 y_ture, y_pred 想象为 A,B 两个集合
# TI = A ∩ B / A ∩ B + α(A-B) + β(B-A) = TP / TP + α·FN + β·FP
# 其中 TP = A ∩ B; FN = A-B; FP = B-A

# α = β = 0.5, 为 Dice
# α = β = 1.0, 为 IoU
# α < β, FP 约束更大
# 所以 TI 是 Dice/IoU 的拓展
def tversky_index(y_true, y_pred, alpha=0.25):
    TP = (y_true * y_pred).sum()
    FN = y_true.sum() - TP
    FP = y_pred.sum() - TP
    return TP / (TP + alpha * FN + (1 - alpha) * FP + epsilon)


def tversky_loss(y_true, y_pred, alpha=0.25):
    return 1 - tversky_index(y_true, y_pred, alpha)


"""Boundary-based"""


# https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
# 计算 fg 到 border 距离倒数 作为 weight
def distance_map_penalty(y_true, y_pred, dist_maps):  # B,C,H,W
    loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    multipled = torch.einsum('bchw,bchw->bchw', loss, dist_maps)  # 即 element-wise; 直接相乘也可
    return multipled.mean()


if __name__ == '__main__':
    # 4个像素点 二分类问题
    y_true = torch.tensor([1.0, 1.0, 0, 0])
    y_pred = torch.tensor([0.9, 0.5, 0.1, 0.4])

    print(dice(y_true, y_pred))
    print(general_dice(y_true, y_pred))
