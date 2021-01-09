from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

def get_data_loader(args):
    file_dataset = "./datasets/csv/dataset_observation_revised_0429_scale10.csv"
    file_itv = "./datasets/csv/dataset_intervention_revised_0429_scale10.csv"

    idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    cv_idx = idx_list[args.cv_idx]
    i = cv_idx[0]
    j = cv_idx[1]

    test_range = range(200 * i, 200 * (i + 1))
    val_range = range(200 * j, 200 * (j + 1))
    train_range = set(range(1000)) - set(test_range) - set(val_range)

    test_idx = np.array(test_range)
    val_idx = np.array(val_range)
    train_idx = np.array(list(train_range))

    data_loader = dict()
    train_data = Collision(file_dataset, file_itv, idx=train_idx)
    valid_data = Collision(file_dataset, file_itv, idx=val_idx)
    test_data = Collision(file_dataset, file_itv, idx=test_idx)

    data_loader['train'] = DataLoader(dataset=train_data, batch_size=args.dataset['batch'],
                                      shuffle=True, collate_fn=gru_regular_time_series)
    data_loader['valid'] = DataLoader(dataset=valid_data, batch_size=args.dataset['batch'],
                                      shuffle=False,collate_fn=gru_regular_time_series)
    data_loader['test'] = DataLoader(dataset=test_data, batch_size=args.dataset['batch'],
                                      shuffle=False,collate_fn=gru_regular_time_series)
    return data_loader

class Collision(Dataset):
    """ DataLoader class for collision datasets"""
    def __init__(self, obs_csv_file, itv_csv_file, idx):
        """
        Args:
            obs_csv_file: path of the observation file
            itv_csv_file: path of the intervention file
        """

        self.positions = pd.read_csv(obs_csv_file)
        self.interventions = pd.read_csv(itv_csv_file)

        # re-initialize index into ID
        self.positions =self.positions.loc[self.positions["ID"].isin(idx)].copy()
        unique_position_id = self.positions["ID"].unique()
        map_dict = dict(zip(unique_position_id, np.arange(len(unique_position_id))))
        self.positions["ID"] = self.positions["ID"].map(map_dict)
        self.positions.set_index("ID", inplace=True)

        self.interventions = self.interventions.loc[self.interventions["ID"].isin(idx)].copy()
        unique_interventions_id = self.interventions["ID"].unique()
        map_dict = dict(zip(unique_interventions_id, np.arange(len(unique_interventions_id))))
        self.interventions["ID"] = self.interventions["ID"].map(map_dict)
        self.interventions.set_index("ID", inplace=True)

        self.length = len(unique_position_id)

    def __len__(self):
        """
        :return: the number of unique ID
        """
        return self.length

    def __getitem__(self, idx):
        obs = self.positions.loc[idx]
        itv = self.interventions.loc[idx]

        return {"idx": idx, "obs":obs, "itv":itv}

def gru_regular_time_series(batch):
    t = torch.tensor(batch[0]["obs"]["Time"].values)
    positions_culumns = [False, True, True, False, False]  #pos_x,pos_y,vel_x,vel_y
    positions = [torch.from_numpy(b["obs"].iloc[:,positions_culumns].values) for b in batch]
    positions = torch.stack(positions, dim=0)   #[batch_size x len(t) x number of features(pos_x, pos_y, vel_x, yel_y)]

    intervention_columns = [False, True, True, True, True]

    mask = [np.logical_and(np.logical_and(b["itv"]["pos_x"]==0., b["itv"]["pos_y"]==0.),
                           np.logical_and(b["itv"]["vel_x"]==0., b["itv"]["vel_y"]==0.))==False for b in batch]

    interventions = [torch.from_numpy(np.c_[b["itv"].iloc[:, intervention_columns].values, m]) for b,m in zip(batch, mask)] #concat mask
    interventions = torch.stack(interventions, dim=0) #[batch_size x len(t) x number of features(vel_x, yel_y)]

    # input_features = torch.cat([positions, interventions], axis=2)
    # vel vector normalization
    res=dict()
    res["t"] = t
    res["obs"] = positions
    res["itv"] = interventions
    # res["feature"] = input_features      # concat of obs and itv
    return res
