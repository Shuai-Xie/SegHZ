import torch
import torch.nn as nn
import torch.nn.functional as F
import constants

"""
print(-math.log(0.01))  # 4.605170185988091     和普通训练基本无异，loss 略小, 1.487 < 1.518 [效果略有提升] 使用 imagenet_pretrain 1.165
print(-math.log(0.05))  # 2.995732273553991     loss 下降偏快，很多没学到, 1.8 -> 0.6
print(-math.log(0.1))  # 2.3025850929940455
print(-math.log(1 / 7))  # 1.9459101490553135

使用纯彩色图片作为训练数据，loss 下降非常快，epoch0 -> 0.9; [最后能训练到 acc->1.0]
因为：
1.标签与数据完全对应
2.每个类的特征很简单，颜色即可区分
3.图像大小不影响 512*512 缩小了一倍，还能分割出细的边界吗? 能!
"""


class CELoss(nn.Module):  # 继承 Module 为了使用 forward 方法
    def __init__(self, weight=None, multi_weights=[1.], ignore_index=constants.BG_INDEX):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
        self.multi_weights = multi_weights

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)  # long 型 target
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)  # set target 1/4 ?

        loss = self.criterion(score, target)
        return loss.mean()

    def forward(self, outputs, target):
        if self.training:
            if len(self.multi_weights) > 1:
                losses = [self._forward(x, target) for x in outputs]
                loss = sum([w * l for (w, l) in zip(self.multi_weights, losses)])
                return loss, losses
            else:
                return self._forward(outputs, target), None
        else:
            return self._forward(outputs, target)  # eval / test


class Thre_CELoss(nn.Module):  # 继承 Module 为了使用 forward 方法
    def __init__(self, weight=None, multi_weights=[1.], ignore_index=constants.BG_INDEX):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
        self.multi_weights = multi_weights
        self.loss_thre = 4.605170185988091  # 抑制噪声; 设置最大, 最小; 让这个阈值也随 epoch 变换?

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)  # long 型 target
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)
        loss[loss > self.loss_thre] = 0.  # 消除掉明显有问题 loss 噪声
        return loss.mean()

    def forward(self, outputs, target):
        if not isinstance(outputs, list):
            outputs = [outputs]
        assert len(self.multi_weights) == len(outputs)
        return sum([w * self._forward(x, target) for (w, x) in zip(self.multi_weights, outputs)])

    def set_loss_thre(self, thre):
        self.loss_thre = thre


class OHEM_CELoss(nn.Module):
    def __init__(self, thresh=0.7, min_kept=20000,
                 weight=None, multi_weights=[1.], ignore_index=constants.BG_INDEX):
        super().__init__()
        self.thresh = thresh
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index
        self.multi_weights = multi_weights
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)
        return loss.mean()  # 注意 reduction='none'

    def _ohem_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_index

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0

        # out[b][1][h][w] = pred[b][ tmp_target[b][1][h][w] ][h][w]
        # 将 target 每个 pixel 指定的 cls_idx 指定的 prob 取出
        pred = pred.gather(1, index=tmp_target.unsqueeze(1))  # B1HW, 和 index size 一致
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]  # min_kept 的 thresh
        thresh = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]  # loss 排序
        pixel_losses = pixel_losses[pred < thresh]  # loss 截取
        return pixel_losses.mean()

    def forward(self, outputs, target):
        if not isinstance(outputs, list):
            outputs = [outputs]
        assert len(self.multi_weights) == len(outputs)

        # 只给末端输出 使用 ohem
        functions = [self._ce_forward] * (len(outputs) - 1) + [self._ohem_forward]

        return sum([
            w * func(x, target)
            for (w, x, func) in zip(self.multi_weights, outputs, functions)
        ])


class LabelSmoothing_CELoss(nn.Module):
    def __init__(self,
                 lbl_smooth=0.1, reduction='mean',
                 weight=None, ignore_index=constants.BG_INDEX):
        super(LabelSmoothing_CELoss, self).__init__()
        self.lbl_smooth = lbl_smooth
        self.reduction = reduction
        self.lbl_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.weight = weight

    @torch.no_grad()
    def get_smooth_label(self, logits, label):
        num_classes = logits.size(1)
        label = label.clone().detach()  # 获取 label data
        ignore_msk = label.eq(self.lbl_ignore)
        label[ignore_msk] = 0  # 非 ignore 保持原样

        lbl_pos, lbl_neg = 1. - self.lbl_smooth, self.lbl_smooth / (num_classes - 1)
        lbl_onehot = torch.empty_like(logits).fill_(lbl_neg).scatter_(1, label.unsqueeze(1), lbl_pos).detach()

        return lbl_onehot, ignore_msk

    def forward(self, logits, label):
        """
        :param logits: B,C,H,W
        :param label: B,H,W
        """
        lbl_onehot, ignore_msk = self.get_smooth_label(logits, label)

        logs = self.log_softmax(logits)  # log(prob) = ori loss
        loss = - torch.sum(logs * lbl_onehot, dim=1)  # B,H,W
        if self.weight is not None:
            loss = loss * self.weight[label]  # cvt weight to label shape, [0, nc-1]

        loss[ignore_msk] = 0

        if self.reduction == 'mean':
            return loss.sum() / ignore_msk.eq(0).sum()  # 非 ignore =1
        if self.reduction == 'sum':
            return loss.sum()


if __name__ == '__main__':
    a = torch.rand(1, 3, 2, 2)
    b = torch.rand(1, 3, 4, 4)
    c = torch.ones(1, 4, 4).long()  # target
    weight = torch.ones(3)

    ohem = OHEM_CELoss(weight=weight, multi_weights=[0.5, 1.])
    print(ohem([a, b], c))
