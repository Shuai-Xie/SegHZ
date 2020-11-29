"""
连续 softmax 两次，使得 loss 比实际值 要高很多
    e^x 是增长率 递增的函数
    先做一次 softmax 将 logits 归一化到 [0,1] probs 后，使得各类别 c 之间 e^c 差异减小，导致正确类的 loss 值升高
"""

import torch.nn.functional as F
import torch

logits = torch.tensor([[5, 1, 2.0]])
target = torch.tensor([0])

loss = F.cross_entropy(logits, target)
print(loss)  # tensor(0.0659)

probs = F.softmax(logits, dim=-1)  # tensor([[0.9362, 0.0171, 0.0466]])
loss = F.cross_entropy(probs, target)
print(loss)  # tensor(0.5932)  loss 提升非常多
