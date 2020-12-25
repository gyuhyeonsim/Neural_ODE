from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

def get_data_loader(args):
    data_loader = dict()
    param_one_series = args.dataset['truncated_one_series']
    train_data = Truncated(args.dataset['trunc_path'],
                           idx=np.arange(600*param_one_series),
                           args=args)  # time - 3
    valid_data = Truncated(args.dataset['trunc_path'],
                           idx=np.arange(600*param_one_series,800*param_one_series),
                           args=args)
    data_loader['train'] = DataLoader(dataset=train_data, batch_size=args.dataset['truncated_batch'],
                                      shuffle=True)
    data_loader['valid'] = DataLoader(dataset=valid_data, batch_size=args.dataset['truncated_batch'],
                                      shuffle=False)
    return data_loader

class Truncated(Dataset):
    """ DataLoader class for collision datasets"""
    def __init__(self, csv_file, idx, args):
        """
        Args:
            obs_csv_file: path of the observation file
            itv_csv_file: path of the intervention file
        """

        obs_path = csv_file+'obs_mask_itv.npy'
        weight_path = csv_file+'SW_CW.npy'

        # folder storing hidden vector of {patient}_{t} -> {idx} as *.npy
        self.hidden_path = csv_file+'hidden/'
        self.obs_mask_itv = torch.from_numpy(np.load(obs_path)[idx,:,:])
        self.cw_sw = torch.from_numpy(np.load(weight_path)[idx,:,:])
        self.args = args
        # print(self.obs_mask_itv.size(),self.cw_sw.size())
        # >> torch.Size([28200, 5, 7]) torch.Size([28200, 5, 2])
        # exit()

    def __getitem__(self, idx):
        input_size = self.args.model['input_size']
        id_t = self.obs_mask_itv[idx,0,:2]
        obs = self.obs_mask_itv[idx,:,2:4]
        mask = self.obs_mask_itv[idx,:,4:6]
        itv = self.obs_mask_itv[idx,:,6:]

        # mask= mask.reshape(-1)
        # for k in mask:
        #     if k<0:
        #         print(k)
        #         print('mask')
        #         print(obs, mask, itv)
        #         exit()

        id = id_t[0].item()
        t = id_t[1].item()
        hidden = np.load(self.hidden_path+'{}_{}.npy'.format(int(id),int(t)-2))
        weight = self.cw_sw[idx]

        return {"idx": idx, "obs": obs, "mask":mask, "itv": itv,
                "weight":weight, 'hidden':hidden}

    def __len__(self):
        # The number of truncated entry
        return self.obs_mask_itv.size(0)