from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import torch

def get_data_loader(args):
    data_loader = dict()
    idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    cv_idx = idx_list[args.cv_idx]
    i = cv_idx[0]
    j = cv_idx[1]
    rand_idx = np.array([89, 36, 210, 181, 147, 70, 69, 13, 139, 28, 41, 210, 36, 95, 202, 117, 18, 248, 215, 198, 116,
                         63, 104, 54, 114, 179, 83, 112, 166, 142, 178, 231, 135, 85, 79, 133, 205, 195, 139, 198, 38,
                         85, 67, 4, 245, 45, 236, 205, 205, 25, 213, 116, 7, 18, 150, 228, 26, 163, 114, 34, 3, 22, 211,
                         199, 28, 211, 110, 47, 197, 132, 104, 71, 202, 180, 82, 169, 249, 5, 172, 88, 156, 133, 153, 2,
                         16, 109, 131, 43, 219, 148, 182, 85, 179, 5, 76, 208, 83, 103, 88, 214, 242, 166, 107, 211,
                         231, 41, 34, 184, 38, 179, 65, 178, 136, 142, 191, 29, 11, 233, 164, 218, 155, 4, 204, 159,
                         118, 159, 71, 35, 178, 78, 144, 237, 68, 20, 114, 137, 239, 149, 157, 17, 178, 171, 93,
                         249, 15, 53, 203, 242, 108, 108, 8, 77, 208, 116, 4, 148, 94, 207, 179, 182, 65, 231, 223,
                         203, 57, 229, 28, 102, 119, 50, 246, 248, 190, 130, 169, 34, 17, 202, 106, 52, 161, 232, 133,
                         78, 197, 94, 59, 42, 237, 193, 245, 216, 151, 125, 13, 79, 221, 153, 25, 54, 19, 165, 107, 171,
                         50, 120, 233, 116, 56, 54, 155, 216, 45, 206, 196, 138, 38, 190, 16, 167, 244, 113, 43, 9, 146,
                         209, 33, 105, 245, 117, 176, 25, 101, 58, 89, 43, 178, 141, 185, 43, 8, 156, 223, 31, 190, 101,
                         107, 145, 210, 164])

    test_range = rand_idx[(50 * i):(50 * (i + 1))]
    val_range = rand_idx[(50 * j):(50 * (j + 1))]
    train_range = np.array(list(set(range(228)) - set(test_range) - set(val_range)))
    file_path = './datasets/csv/final_patient_30_std.csv'

    data_train = patient_traj(data_path=file_path, idx=train_range)
    data_val = patient_traj(data_path=file_path, idx=val_range)
    data_test = patient_traj(data_path=file_path, idx=test_range)

    data_loader['train'] = DataLoader(dataset=data_train, shuffle=True, collate_fn=regular_time_series,
                          batch_size=args.batch_size)
    data_loader['valid'] = DataLoader(dataset=data_val, shuffle=False, collate_fn=regular_time_series,
                        batch_size=len(data_val))
    data_loader['test'] = DataLoader(dataset=data_test, shuffle=False, collate_fn=regular_time_series,
                         batch_size=len(data_test))
    return data_loader

class patient_traj(Dataset):
    def __init__(self, data_path, idx):
        self.all_data = pd.read_csv(data_path)

        self.positions = self.all_data[['ID', 'Time', 'Systolic_BP', 'Diastolic_BP', 'Mean_BP']]
        self.interventions = self.all_data[['ID', 'Time', 'norepinephrine', 'vasopressin',
                                            'propofol', 'amiodarone', 'phenylephrine']]

        # re-initialize index into ID
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


def regular_time_series(batch):
    t = torch.tensor(batch[0]["obs"]["Time"].values)
    positions_culumns = [False, True, True, True]
    positions = [torch.from_numpy(b["obs"].iloc[:, positions_culumns].values) for b in batch]
    positions = torch.stack(positions, dim=0)  # [batch_size x len(t) x number of features(pos_x, pos_y, vel_x, yel_y)]

    intervention_columns = [False, True, True, True, True, True]
    mask = [np.logical_and(np.logical_and(b["itv"]["norepinephrine"]==0., b["itv"]["vasopressin"]==0.),
            np.logical_and((np.logical_and(b["itv"]["propofol"]==0.,  b["itv"]["amiodarone"]==0.)), b["itv"]["phenylephrine"]==0.))==False for b in batch]

    interventions = [torch.from_numpy(np.c_[b["itv"].iloc[:, intervention_columns].values,m]) for b,m in zip(batch, mask)]
    interventions = torch.stack(interventions, dim=0)  # [batch_size x len(t) x number of features(vel_x, yel_y)]

    res = dict()
    res["t"] = t
    res["obs"] = positions
    res["itv"] = interventions

    return res