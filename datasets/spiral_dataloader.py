"""
I follows the ode demo of ricky; https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py
"""

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data

from torchdiffeq import odeint



def get_batch(args, data_size, batch_time, batch_size):
    # data size 50
    # batch_time 50

    true_y0 = torch.tensor([[2., 0.]]).to(args.device)
    t = torch.linspace(0., 25., data_size).to(args.device)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(args.device)

    class Lambda(nn.Module):
        def forward(self, t, y):
            return torch.mm(y ** 3, true_A)

    with torch.no_grad():
        true_y = odeint(Lambda(), true_y0, t, method='dopri5')

    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))

    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(args.device), batch_t.to(args.device), batch_y.to(args.device)

class Spiral(data.Dataset):
    def __init__(self, args, torch_data):
        self.datasets = torch_data

    def __len__(self):
        return self.datasets.size(0)

    def __getitem__(self, idx):
        entry_dict = dict()
        entry_dict['init_point'] = self.datasets[0][idx]
        entry_dict['time_point'] = self.datasets[1]
        entry_dict['traj'] = self.datasets[2][:,idx,:,:]
        return entry_dict

def get_data_loader(args):
    data_size = 1000
    batch_time = 10
    train_batch_size = 60
    valid_batch_size = 20
    test_batch_size = 20

    train_datasets = get_batch(args, data_size, batch_time, batch_size=train_batch_size)
    valid_datasets = get_batch(args, data_size, batch_time, batch_size=valid_batch_size)
    test_datasets = get_batch(args, data_size, batch_time, batch_size=test_batch_size)

    # print(train_datasets[0].size(),train_datasets[1].size(), train_datasets[2].size())
    # >> torch.Size([60, 1, 2]) torch.Size([10]) torch.Size([10, 60, 1, 2])
    data_loader = dict()
    data_loader['train'] = Spiral(args, train_datasets)
    data_loader['valid'] = Spiral(args, valid_datasets)
    data_loader['test'] = Spiral(args, test_datasets)
    return data_loader