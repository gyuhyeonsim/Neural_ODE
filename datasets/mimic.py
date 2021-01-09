from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import torch
def get_data_loader(args):
    data_loader = dict()
    idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    data_idx_num = idx_list[args.cv_idx]
    i = data_idx_num[0]
    j = data_idx_num[1]
    rand_idx = np.load('./datasets/csv/mimic_index.npy')
    data_path = './datasets/csv/mimic_patient.csv'

    test_range = rand_idx[(50 * i):(50 * (i + 1))]
    val_range = rand_idx[(50 * j):(50 * (j + 1))]
    train_range = np.array(list(set(range(263)) - set(test_range) - set(val_range)))

    data_train = mimic_dataset(data_path=data_path, idx=train_range)
    data_val = mimic_dataset(data_path=data_path, idx=val_range)
    data_test = mimic_dataset(data_path=data_path, idx=test_range)

    data_loader['train'] = DataLoader(dataset=data_train, shuffle=True, collate_fn=mimic_collate,
                          batch_size=args.batch_size)
    data_loader['valid'] = DataLoader(dataset=data_val, shuffle=False, collate_fn=mimic_collate,
                        batch_size=len(data_val))
    data_loader['test'] = DataLoader(dataset=data_test, shuffle=False, collate_fn=mimic_collate,
                         batch_size=len(data_test))

    return data_loader


class mimic_dataset(Dataset):
    def __init__(self, data_path, idx):
        self.all_data = pd.read_csv(data_path)

        self.positions = self.all_data[['ID', 'Time', 'Systolic_BP', 'Diastolic_BP', 'Mean_BP']]
        self.interventions = self.all_data[['ID', 'Time', 'hydralazine', 'diltiazem', 'furosemide',
                                            'phenylephrine', 'metoprolol', 'epinephrine']]

        self.positions = self.positions.loc[self.positions["ID"].isin(idx)].copy()
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

        return {"idx": idx, "obs": obs, "itv": itv}


def mimic_collate(batch):
    t = torch.tensor(batch[0]["obs"]["Time"].values)
    positions_culumns = [False, True, True, True]
    positions = [torch.from_numpy(b["obs"].iloc[:, positions_culumns].values) for b in batch]
    positions = torch.stack(positions, dim=0)

    # concatenate censoring inormation
    intervention_columns = [False, True, True, True, True, True, True]
    mask = [np.logical_and(np.logical_and(np.logical_and(b["itv"]["hydralazine"]==0., b["itv"]["diltiazem"]==0.),
            np.logical_and(b["itv"]["furosemide"]==0.,  b["itv"]["phenylephrine"]==0.)),
            np.logical_and(b["itv"]["metoprolol"]==0.,  b["itv"]["epinephrine"]==0.))==False for b in batch]

    # np.c_[b["itv"].iloc[:, intervention_columns].values, m]
    interventions = [torch.from_numpy(np.c_[b["itv"].iloc[:, intervention_columns].values,m])
                     for b,m in zip(batch,mask)]
    interventions = torch.stack(interventions, dim=0)

    res = dict()
    res["t"] = t
    res["obs"] = positions
    res["itv"] = interventions

    return res