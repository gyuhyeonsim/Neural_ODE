import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch

def get_data_loader(args):
    data_loader = dict()
    train_data = Sinusoid_from_scratch(mode='train')
    # train_data = Sinusoid(args.dataset['path'], mode='train')
    # valid_data = Sinusoid(args.dataset['path'], mode='valid')
    data_loader['train'] = DataLoader(dataset=train_data, batch_size=args.dataset['batch'], shuffle=True)
    # data_loader['valid'] = DataLoader(dataset=valid_data, batch_size=args.dataset['batch'], shuffle=False)
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


class Sinusoid_from_scratch(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.samp_sinusoidals, self.samp_ts, self.amps = self.sinusoid_from_scratch(n_sinusoid= 1024, n_total=2000, n_sample=400, skip_step=4)
        elif mode == 'valid':
            pass
        elif mode == 'test':
            pass

        self.len = self.samp_sinusoidals.size(0)
        print('[Dataset] Sinusoid_from_scratch dataset for {} is initialized'.format(mode))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        samp_sin = self.samp_sinusoidals[item]
        samp_ts = self.samp_sinusoidals[item]
        amp = self.amps[item]
        dict_to_return = {'point': samp_sin, 't': samp_ts, 'amp':amp}
        return dict_to_return


    def sinusoid_from_scratch(self, n_sinusoid = 1024, n_total=2000, n_sample=400, skip_step=4):
        start = 0.
        stop = 6. * np.pi
        orig_ts = np.linspace(start, stop, num=n_total)
        samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

        samp_sinusoidals = []
        count = n_sinusoid / 4
        amp1 = [1] * count + [2] * count + [3] * count + [4] * count
        count = n_sinusoid / 8
        amp2 = ([1] * count + [2] * count + [3] * count + [4] * count) * 2
        count = n_sinusoid / 16
        amp3 = ([1] * count + [2] * count + [3] * count + [4] * count) * 4
        count = n_sinusoid / 32
        amp4 = ([1] * count + [2] * count + [3] * count + [4] * count) * 8

        amps = np.stack((amp1, amp2, amp3, amp4), axis=1)
        assert len(amps) == n_sinusoid, "amp list incorrectly constructed!"

        for i in range(n_sinusoid):
            amp = amps[i]
            samp_sinusoidal = -amp[0] * np.cos(samp_ts) -amp[1] * 0.5 * np.cos(2 * samp_ts) + amp[2] * np.sin(samp_ts) + amp[3] * 0.5 * np.sin(2*samp_ts)
            samp_sinusoidals.append(samp_sinusoidal)

        samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
        samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)  # batch_size x seq_len x 1
        amps = torch.Tensor(amps)  # batch_size x 4
        samp_ts = torch.Tensor([samp_ts] * n_sinusoid)   # n_sinusoid x 400
        return samp_sinusoidals, samp_ts, amps








