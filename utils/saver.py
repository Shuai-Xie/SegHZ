import os
import shutil
import torch
import json
import numpy as np
import constants
from utils import dump_pickle

"""
experiment_dir: tensorboard
    save/load checkpoint
    save_experiment_config
    save_active_selections, mask of select parts
"""


class Saver:

    def __init__(self, args, suffix='', timestamp='',
                 experiment_group=None, remove_existing=False):

        self.args = args

        if experiment_group is None:
            experiment_group = args.dataset

        # runs/ tensorboard
        self.experiment_dir = os.path.join(constants.RUNS, experiment_group,
                                           f'{args.checkname}_{timestamp}', suffix)

        if remove_existing and os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)

        if not os.path.exists(self.experiment_dir):
            print(f'Creating dir {self.experiment_dir}')
            os.makedirs(self.experiment_dir)

    def save_test_results(self, data, filename='test_results.pkl'):
        filename = os.path.join(self.experiment_dir, filename)
        dump_pickle(data, filename)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename='checkpoint.pth.tar', file_path=None):
        filename = os.path.join(self.experiment_dir, filename) if not file_path else file_path
        return torch.load(filename)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        arg_dictionary = vars(self.args)
        log_file.write(json.dumps(arg_dictionary, indent=4, sort_keys=True))  # 按 key 排序
        log_file.close()
