import numpy as np


def hz_map_size():
    """必然为 256 整数倍
    z16 [16384 15616]
    z18 [65536 62464]
    """
    w, h = 11153, 10532
    target_size = np.array([w, h])

    # z16
    ratio = 512 / 350
    z16_size = target_size * ratio
    z16_size = (z16_size // 256 + 1) * 256
    z18_size = z16_size * 4

    return z16_size.astype(int), z18_size.astype(int)


def crop_tiles_size(whole_size, base_multiples):
    """
    z16 [16384 15616] -> 256 对应 z18 1024
        256: 64 x 61 = 3904
        512: 32 x 30 = 960
        768: 21 x 20 = 420
        1024: 16 x 15 = 240
    z18 [65536 62464]
        256: 256 x 244 = 62464
        512: 128 x 122 = 15616
        768: 85 x 81 = 6885
        1024: 64 x 61 = 3904
    """
    print(whole_size)
    for mutli in base_multiples:
        size = 256 * mutli
        crop_w, crop_h = whole_size // size
        print('{}: {} x {} = {}'.format(size, crop_w, crop_h, crop_h * crop_w))


if __name__ == '__main__':
    z16_size, z18_size, = hz_map_size()
    crop_tiles_size(z16_size, range(1, 5))
    crop_tiles_size(z18_size, range(1, 5))
