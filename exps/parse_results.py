import torch
import os
from pprint import pprint


def load_eval_results(exp):
    state = torch.load(f'runs/HZ_Merge/{exp}/checkpoint.pth.tar')
    state.pop('state_dict')
    pprint(state)


def runs_dir_evals():
    runs_dir = 'runs/HZ_Merge'

    for exp in os.listdir(runs_dir):
        state = torch.load(f'{runs_dir}/{exp}/checkpoint.pth.tar')
        state.pop('state_dict')
        print(exp)
        pprint(state)


if __name__ == '__main__':
    runs_dir_evals()
