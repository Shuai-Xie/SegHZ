import torch
from torch.nn import Module, Parameter, Softmax, Conv2d


class PAM(Module):
    """ Position attention module"""

    def __init__(self, in_dim, reduction=8):
        super(PAM, self).__init__()
        hidden_dim = in_dim // reduction

        # features 降维，计算 attention matrix
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1)  # 压缩为 hidden_dim
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 选择是否压缩

        self.gamma = Parameter(torch.zeros(1))  # learned attention scale factor
        self.softmax = Softmax(dim=-1)  # 对 HxW 维归一化，即每行 weights sum = 1

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)

            HxW 不宜太大，不然内存直接崩掉
        """
        B, _, H, W = x.size()

        # self-attention 机制，空间计算 pixel 相似度, 两个 conv 分别计算 query, key
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # B reshape & transpose (B,H*W,C)
        proj_key = self.key_conv(x).view(B, -1, W * H)  # C reshape (B,C,H*W)
        energy = torch.bmm(proj_query, proj_key)  # batch matrix multiplication, (B, H*W, H*W)
        attention = self.softmax(energy)  # S softmax 归一化 最后一维 (B, H*W, H*W), 两个 softmax 向量方向越一致，数量积越大
        proj_value = self.value_conv(x).view(B, -1, W * H)  # D reshape (B, out_dim, H*W)

        # D * S, 单点值 = 全部点 attention 之和
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B,C,H*W)
        out = out.view(B, -1, H, W)  # new attentioned features

        # 要 +x 因为 gamma 初始化为 0
        out = self.gamma * out + x

        return out


if __name__ == '__main__':
    m = PAM(40, 3)

    for name, param in m.named_parameters():  # 递归寻找 Parameter 类型, 可学习 param
        print(name, param.shape)

    """
    gamma               torch.Size([1])
    query_conv.weight   torch.Size([3, 10, 1, 1])
    query_conv.bias     torch.Size([3])
    key_conv.weight     torch.Size([3, 10, 1, 1])
    key_conv.bias       torch.Size([3])
    value_conv.weight   torch.Size([10, 10, 1, 1])
    value_conv.bias     torch.Size([10])
    """
