import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

def get_dataloader(args):
    train_data = Sinusoid_from_scratch(dataset_type=args.dataset['type'])
    data_loader = DataLoader(dataset=train_data, batch_size=args.dataset['batch'], shuffle=True)
    return data_loader




def dataset1(n_sinusoid=1024, n_total=2000, n_sample=400, skip_step=4):
    """
    √3cos(𝑥−0.615) =  √2cos(𝑥)+sin(𝑥)
    samp_sinusoidals.shape = 2000 x 400 x 1
    samp_ts.shape = 2000 x 400
    amps.shape = 2000 x 1
    """
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    sinusoidals = []
    samp_sinusoidals = []
    amps = []

    for i in range(0, n_sinusoid):
        if i < 500:
            amp = 1
        elif i < 1000:
            amp = 2
        elif i < 1500:
            amp = 3
        else:
            amp = 4

        sinusoidal = amp * (np.sqrt(3) * np.cos(orig_ts - 0.615))
        sinusoidals.append(sinusoidal)

        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)

        amps.append(amp)

    sinusoidals = np.stack(sinusoidals, axis=0)
    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = np.stack(amps, axis=0)
    amps = torch.unsqueeze(torch.Tensor(amps), dim=-1)
    samp_ts = torch.Tensor([samp_ts] * 2000)

    return samp_sinusoidals, samp_ts, amps



def dataset2(n_sinusoid=2000, n_total=2000, n_sample=400, skip_step=4):
    """
    𝑦(𝑥)=sin(2𝑥)−cos(𝑥)
    samp_sinusoidals.shape = 2000 x 400 x 1
    samp_ts.shape = 2000 x 400
    amps.shape = 2000 x 1
    """
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    sinusoidals = []
    samp_sinusoidals = []
    amps = []

    for i in range(0, n_sinusoid):
        if i < 500:
            amp = 1
        elif i < 1000:
            amp = 2
        elif i < 1500:
            amp = 3
        else:
            amp = 4

        sinusoidal = amp * (np.sin(2 * orig_ts) - np.cos(orig_ts))
        sinusoidals.append(sinusoidal)

        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)

        amps.append(amp)

    sinusoidals = np.stack(sinusoidals, axis=0)
    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = np.stack(amps, axis=0)
    amps = torch.unsqueeze(torch.Tensor(amps), dim=-1)
    samp_ts = torch.Tensor([samp_ts] * 2000)

    return sinusoidals, orig_ts, samp_sinusoidals, samp_ts, amps



def dataset3(n_sinusoid=2000, n_total=2000, n_sample=400, skip_step=4):
    """
    𝑦(𝑥)=−4 sin(𝑥)+sin(2𝑥)−cos(𝑥)+0.5 cos(2𝑥)
    samp_sinusoidals.shape = 2000 x 400 x 1
    samp_ts.shape = 2000 x 400
    amps.shape = 2000 x 1
    """
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    sinusoidals = []
    samp_sinusoidals = []
    amps = []

    for i in range(0, n_sinusoid):
        if i < 500:
            amp = 1
        elif i < 1000:
            amp = 2
        elif i < 1500:
            amp = 3
        else:
            amp = 4

        sinusoidal = amp * (-4 * np.sin(orig_ts) + np.sin(2 * orig_ts) - np.cos(orig_ts) + 0.5 * np.cos(2 * orig_ts))
        sinusoidals.append(sinusoidal)

        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)

        amps.append(amp)

    sinusoidals = np.stack(sinusoidals, axis=0)
    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = np.stack(amps, axis=0)
    amps = torch.unsqueeze(torch.Tensor(amps), dim=-1)
    samp_ts = torch.Tensor([samp_ts] * 2000)

    return sinusoidals, orig_ts, samp_sinusoidals, samp_ts, amps

def dataset4(n_sinusoid=1024, n_total=2000, n_sample=300, skip_step=4):
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    count = int(n_sinusoid / 4)
    amp1 = [1] * count + [2] * count + [3] * count + [4] * count
    count = int(n_sinusoid / 8)
    amp2 = ([1] * count + [2] * count + [3] * count + [4] * count) * 2
    count = int(n_sinusoid / 16)
    amp3 = ([1] * count + [2] * count + [3] * count + [4] * count) * 4
    count = int(n_sinusoid / 32)
    amp4 = ([1] * count + [2] * count + [3] * count + [4] * count) * 8

    amps = np.stack((amp1, amp2, amp3, amp4), axis=1)
    assert len(amps) == n_sinusoid, "amp list incorrectly constructed!"

    for i in range(n_sinusoid):
        amp = amps[i]
        samp_sinusoidal = -amp[0] * np.cos(samp_ts) - amp[1] * 0.5 * np.cos(2 * samp_ts) + amp[2] * np.sin(samp_ts) + \
                          amp[3] * 0.5 * np.sin(2 * samp_ts)
        samp_sinusoidals.append(samp_sinusoidal)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)  # batch_size x seq_len x 1
    amps = torch.Tensor(amps)  # batch_size x 4
    samp_ts = torch.Tensor([samp_ts] * n_sinusoid)  # n_sinusoid x 400
    return samp_sinusoidals, samp_ts, amps


class Sinusoid_from_scratch(Dataset):
    def __init__(self, dataset_type):
        if dataset_type == 'dataset1':
            dataset_type = dataset1
        self.samp_sin, self.samp_ts, self.latent_v = dataset_type()

    def __len__(self):
        return self.samp_sin.size(0)

    def __getitem__(self, item):
        samp = self.samp_sin[item]
        samp_ts = self.samp_ts[item]
        latent_v = self.latent_v[item]
        return samp, samp_ts, latent_v

