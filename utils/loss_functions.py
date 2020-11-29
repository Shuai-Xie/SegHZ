import torch
import torch.nn.functional as F

epsilon = 1e-5


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


def dice_coef(y_true, y_pred):  # B,C,H,W
    y_true = flatten(y_true)  # C,N
    y_pred = flatten(y_pred)  # C,N
