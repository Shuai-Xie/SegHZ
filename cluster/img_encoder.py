import torch
import torch.nn as nn
from models.deeplab.backbone.mobilenet import MobileNetV2
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np


class HZDataset(Dataset):
    def __init__(self, root, base_size):
        self.img_paths = [os.path.join(root, p) for p in sorted(os.listdir(root))]
        self.transform = transforms.Compose([
            transforms.Resize(base_size),  # 默认指定 smaller edge
            transforms.ToTensor(),  # Normalize tensor image
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transform(img)
        return img


device = torch.device('cuda:1')


@torch.no_grad()
def encode_image():
    dataset = HZDataset(root='/nfs2/sontal/RS_images/16', base_size=224)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    feats = None

    def hook(module, input, output):  # return 会将结果 接着网络计算，所以不能轻易修改 output
        nonlocal feats
        feats = output

    model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True).to(device).eval()
    # high_level features 注册 hook, 方便获取中间特征
    model.high_level_features.register_forward_hook(hook)

    res = torch.zeros((0, 320)).to(device)
    for imgs in tqdm(dataloader):
        imgs = imgs.to(device)
        model(imgs)
        feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        res = torch.cat((res, feats))

    np.save('cluster/hz_features.npy', res.cpu().numpy())


if __name__ == '__main__':
    encode_image()
    pass
