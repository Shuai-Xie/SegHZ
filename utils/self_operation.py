import torch
import random


class SelfOperation:

    def __init__(self, opt, *args, **kwargs):
        self.opt = opt
        self.args = args
        self.kwargs = kwargs
        self.funcs = None
        self.inv_funcs = None

    def op(self, imgs):  # B,C,H,W
        self.flip_degree = random.randint(1, 3)  # {1,2,3} 90,180,270 逆时针
        inv_imgs = torch.rot90(imgs, k=self.flip_degree, dims=(2, 3))  # k 表示旋转次数
        return inv_imgs

    def inv_op(self, labels):
        return torch.rot90(labels, 4 - self.flip_degree, dims=(2, 3))

    # apply additional operation for self-supervised training
    def apply(self, func, inv_func, *args, **kwargs):
        self.op = func
        self.inv_op = inv_func

    def __call__(self, batch, inv=False, *args, **kwargs):
        if inv:
            return self.inv_op(batch)
        else:
            return self.op(batch)
