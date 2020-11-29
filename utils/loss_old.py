import torch
import torch.nn as nn
import torch.nn.functional as F
import constants


def focal_loss_sigmoid(y_pred, labels, alpha=0.25, gamma=2):
    labels = labels.float()

    # loss = label1 + label0
    # 难易样本占比，1- y_pred 约束, y_pred 越高，越容易
    loss = -labels * (1 - alpha) * ((1 - y_pred) ** gamma) * torch.log(y_pred) - \
           (1 - labels) * alpha * (y_pred ** gamma) * torch.log(1 - y_pred)

    return loss


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W) 将 C 通道带出
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)  # C,N


def onehot(tensor, num_class, device):
    B, H, W = tensor.shape
    y = torch.zeros(num_class, B, H, W, requires_grad=False).to(device)

    for i in range(num_class):
        y[i][tensor == i] = 1  # 自动过滤掉 bg_idx
    return y.permute(1, 0, 2, 3)  # B,C,H,W


class SegmentationLosses:

    def __init__(self, loss_type='ce', weight=None, ignore_index=constants.BG_INDEX,
                 device='cpu', multi_output=False, multi_weights=None):
        self.ignore_index = ignore_index
        self.weight = weight
        self.device = device
        self.epsilon = 1e-5

        self.loss_fn = self.get_loss_function(loss_type)
        if multi_output:  # loss_fn 作为参数 输入 func
            self.loss_fn = self.multi_output_loss(self.loss_fn, loss_type)
        self.multi_weights = multi_weights

    def __call__(self, output, target):
        return self.loss_fn(output, target)

    def get_loss_function(self, loss_type):
        if loss_type == 'ce':
            return self.CELoss
        elif loss_type == 'bce':
            return self.BCELoss
        elif loss_type == 'focal':
            return self.FocalLoss
        elif loss_type == 'dice':
            return self.DiceLoss
        elif loss_type == 'dice_ce':
            return self.Dice_CE_Loss
        else:
            raise NotImplementedError

    def multi_output_loss(self, loss_fn, loss_type):
        # 返回 loss_fn 实例化的 对应函数
        def inner_loss(outputs, target):
            _, h, w = target.shape
            if not isinstance(outputs, list):
                outputs = [outputs]

            # one-hot target
            if loss_type != 'ce':
                target = onehot(target, num_class=outputs[0].size(1), device=self.device)

            loss = 0.
            if not self.multi_weights:
                self.multi_weights = [1.] * len(outputs)
            for i, out in enumerate(outputs):
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
                loss += loss_fn(out, target) * self.multi_weights[i]
            return loss

        return inner_loss

    def CELoss(self, output, target):
        """
        @param output: [B,C,H,W]    模型的输出；不需要 softmax, CELoss 内部会完成
        @param target: [B,H,W]
        """
        # logpt [B,H,W] -> mean
        return F.cross_entropy(output, target.long(),
                               weight=self.weight,  # weigh of each class
                               ignore_index=self.ignore_index,
                               reduction='mean')

    def BCELoss(self, output, target, use_pos_weight=False):
        """
         @param output: [B,C,H,W]    模型的输出；不需要 softmax, binary_cross_entropy_with_logits 会完成
         @param target: [B,C,H,W]
         """
        # B,C,H,W
        output, target = flatten(output), flatten(target)  # N,C
        pos_weight = None
        if use_pos_weight:
            pos_cnts = target.sum(dim=0)  # C, cnt of each cls
            pos_weight = (target.shape[0] - pos_cnts) / (pos_cnts + 1e-5)

        # 将 BCE 与 sigmoid 合成一步 https://blog.csdn.net/qq_22210253/article/details/85222093
        loss = F.binary_cross_entropy_with_logits(output.t(), target.t(),  # N,C
                                                  pos_weight=pos_weight,
                                                  reduction='none')
        loss = torch.mean(self.weight * loss.mean(0))
        return loss

    def FocalLoss(self, output, target, gamma=2, alpha=1):
        """
        @param output: [B,C,H,W]    模型的输出；不需要 softmax, CELoss 内部会完成
        @param target: [B,C,H,W]
        @param gamma: hard-easy regulatory factor 调节难易样本的抑制
        @param alpha: class imbalance regulatory factor 定义正样本的权值, CE 只用了正样本; 作者实验中正例的 alpha 反而小的
        """
        # CE 只计算了正类的 logpt，不符合 focal loss
        # BCE, [B,C,H,W] -> -log(pt), 正例/负例 概率 loss
        output, target = flatten(output), flatten(target)
        bce_loss = F.binary_cross_entropy_with_logits(output.t(), target.t(), reduction='none')  # N,C
        # loss = -log(pt)
        # pt = e^(-loss)
        pt = torch.exp(-bce_loss)
        # if p=0.9, (1-p)^2, loss 小 100 倍
        # if p=0.5, loss 只小 4 倍
        # 正负样本都是 此形式
        loss = alpha * (1 - pt) ** gamma * bce_loss  # C,N
        loss = torch.mean(self.weight * loss.mean(-1))  # focal loss 基础上，对每类加权
        return loss

    def IoULoss(self, output, target):
        output, target = flatten(output).sigmoid(), flatten(target)  # C,N
        inter = (output * target).sum(-1)  # C
        union = (output + target).sum(-1)  # C
        loss = 1 - (inter / union.clamp(min=self.epsilon)).mean()
        return loss

    def DiceLoss(self, output, target):
        # sigmoid 两类分别计算
        output, target = flatten(output).sigmoid(), flatten(target)  # C,N
        inter = (output * target).sum(-1)  # C
        union = (output ** 2 + target ** 2).sum(-1)  # C
        loss = 1 - (2 * inter / union.clamp(min=self.epsilon)).mean()
        return loss

    def GDiceLoss(self, output, target):
        """
        Generalized Dice loss 多分类 dice loss
        @param output: [B,C,H,W]    模型输出 do softmax, must sum to 1 over C channel
        @param target: [B,C,H,W]    one-hot label
        """
        # output = F.softmax(output, dim=1)  # 多类 prob
        output, target = flatten(output).sigmoid(), flatten(target)  # 计算二分类 loss

        target_sum = target.sum(-1)  # 各类数量 C
        # todo: 动态 weights or 静态? 静态无法避免一些为空的类
        # https://stats.stackexchange.com/questions/414358/why-are-weights-being-used-in-generalized-dice-loss-and-why-cant-i
        cls_weights = 1. / (target_sum * target_sum).clamp(min=self.epsilon)  # 空类 1/eps

        # mask = target_sum.clone().gt(0)  # 选中有效的类 loss 才开始下降; 不然 loss 始终 = 1
        # cls_weights = cls_weights * mask

        # 各类交集, target 中 空类 inter 自然 = 0
        # 归一化, sum 求比值
        inter = (output * target).sum(-1) * cls_weights
        union = (output + target).sum(-1) * cls_weights

        loss = 1 - 2 * inter.sum() / union.sum().clamp(min=self.epsilon)
        return loss

    def Dice_CE_Loss(self, output, target):
        return self.DiceLoss(output, target) + self.CELoss(output, target.argmax(dim=1))


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

        if sigmoid_normalization:  # two class
            self.normalization = nn.Sigmoid()
        else:  # multi
            self.normalization = nn.Softmax(dim=1)  # B,C,H,W

    def forward(self, input, target):
        # input -> probs
        input = self.normalization(input)  # N,C
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)  # not equal
            mask.requires_grad = False  # binary
            # 忽略部分 都=0
            input = input * mask
            target = target * mask

        input = flatten(input)  # todo: C,N
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)  # 每类的数量
        class_weights = 1. / (target_sum * target_sum).clamp(min=self.epsilon)  # weight of each c

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            intersect = self.weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


class OHEM_CrossEntroy_Loss(nn.Module):
    def __init__(self, threshold, keep_num):
        super(OHEM_CrossEntroy_Loss, self).__init__()
        self.threshold = threshold
        self.keep_num = keep_num
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        # OHEM, _ = loss.topk(k=int(OHEM_percent * [*loss.shape][0]))
        loss = self.loss_function(output, target).view(-1)
        loss, loss_index = torch.sort(loss, descending=True)
        threshold_in_keep_num = loss[self.keep_num]
        if threshold_in_keep_num > self.threshold:
            loss = loss[loss > self.threshold]
        else:
            loss = loss[:self.keep_num]  # 保存部分 hard example 训练
        return torch.mean(loss)


if __name__ == "__main__":
    a = torch.rand(1, 3, 2, 2)
    b = torch.rand(1, 3, 4, 4)
    c = torch.ones(1, 4, 4)  # target
    weight = torch.ones(3)
    # weight = None

    f = SegmentationLosses('dice', weight, multi_output=True, multi_weights=[1., 1.])
    loss = f([a, b], c)
    # loss = f(a, c)
    print(loss)
