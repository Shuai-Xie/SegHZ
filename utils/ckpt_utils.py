import torch


def save_model_from_ckpt():
    ckpt_path = 'runs/HZ_Merge/hrnet_green_Sep22_104053/checkpoint.pth.tar'
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    state_dict = ckpt['state_dict']
    torch.save(state_dict, 'hrnetv2_w48_sat_pretrained.pth')


def check_pretrain():
    # state_dict = torch.load('pretrained_models/hrnetv2_w48_imagenet_pretrained.pth')
    state_dict = torch.load('pretrained_models/hrnetv2_w48_sat_pretrained.pth')
    for k, v in state_dict.items():
        print(k, v.shape)
