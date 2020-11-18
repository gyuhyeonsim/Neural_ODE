import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch

def get_data_loader(args):
    data_loader = dict()
    train_data = Sinusoid(args.dataset['path'], mode='train')
    valid_data = Sinusoid(args.dataset['path'], mode='valid')
    data_loader['train'] = DataLoader(dataset=train_data, batch_size=args.dataset['batch'], shuffle=True)
    data_loader['valid'] = DataLoader(dataset=valid_data, batch_size=args.dataset['batch'], shuffle=False)
    return data_loader

class Sinusoid(Dataset):
    def __init__(self, path, mode='train'):
        dataset = pd.read_csv(path)
        self.mode = mode
        if mode =='train':
            self.dataset = dataset.loc[dataset['id']<600]
        elif mode =='valid':
            self.dataset = dataset.loc[(dataset['id']>=600) & (dataset['id']<800)]
        elif mode =='test':
            pass

        self.len = len(np.unique(self.dataset['id']))
        print('[Dataset] Sinusoid dataset for {} is initialized'.format(mode))

    def __getitem__(self, idx):
        if self.mode=='valid':
            idx+=600
        data = self.dataset.loc[self.dataset['id']==idx]
        t = data['t'].to_numpy()
        point = data['point'].to_numpy()
        freq = data['freq'].to_numpy()[0]
        amp = data['amp'].to_numpy()[0]
        sign = data['sign'].to_numpy()[0]
        dict_to_return = {'t': t, 'point': point, 'freq': freq, 'amp': amp, 'sign':sign}
        for key in list(dict_to_return.keys()):
            if key =='freq' or key =='amp' or key =='sign':
                continue
            dict_to_return[key] = torch.from_numpy(dict_to_return[key])
        dict_to_return['point'] = dict_to_return['point'].unsqueeze(1)
        return dict_to_return

    def __len__(self):
        return self.len