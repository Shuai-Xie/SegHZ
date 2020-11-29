import pickle
import os
import numpy as np


def save_ndarray_to_binary(ndarray: np.ndarray, path):
    ndarray = ndarray.astype(np.uint16)
    ndarray.byteswap(True)
    ndarray.tofile(path)
    index_path = path.replace('.b', '.txt')
    with open(index_path, 'w') as f:
        str = '{} {} {} {} {} {}'.format(path.split('/')[-1], 0, ndarray.shape[1], 0, ndarray.shape[0], 1)
        f.write(str)


def transfer(pkl_path, binary_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            save_ndarray_to_binary(data[key]['pred'],
                                   os.path.join(binary_path, '{}.b'.format(key)))


if __name__ == '__main__':
    transfer("data/test/hrnet_ce_Sep10_230343/result.pkl", "data/test/bin")
