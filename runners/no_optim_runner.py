import os

import torch
import numpy as np
import torch.nn as nn
import random
import pandas as pd
import time
import os
import random

from torch.utils.tensorboard import SummaryWriter
from utils.run_utils.tensorboard_utils import *

class NoOptimRunner():
    def __init__(self, args, dataloader, model, optim):
        super().__init__()
        self.noise_std = .3
        self.args = args
        self.model = model
        self.dl = dataloader
        self.optim = None
        self.seed = random.randint(0, 10000)
        self.cal_loss = nn.MSELoss()
        self.writer = SummaryWriter("./runs/{}".format(self.args.exid))

    def train(self):
        # no device used
        total_iloss = 0
        total_eloss = 0
        total_inum = 0
        total_enum = 0
        total_time = 0

        imse_list = []
        emse_list = []

        for j, b in enumerate(self.dl['valid']):
            t = b['t'][0].float()[:self.args.dataset['interpolation']]  # all data entries share the same time point
            point = b['point'].float()[:, :self.args.dataset['interpolation']]
            point = point.squeeze(0).squeeze(1).numpy()

            extra_t = b['t'][0].float()
            extra_point = b['point'].float()[:, self.args.dataset['interpolation']:]
            extra_point = extra_point.squeeze(0).squeeze(1)

            start_t = time.time()
            pred, extra_pred = self.model(t, extra_t, point)
            burden_t = time.time()-start_t
            total_time+=burden_t

            point = torch.from_numpy(point)
            extra_pred = extra_pred[self.args.dataset['interpolation']:]

            instant_iloss = self.cal_loss(point, pred).item()
            instant_eloss = self.cal_loss(extra_point, extra_pred).item()

            total_inum+=1
            total_enum+=1

            total_iloss+=instant_iloss
            total_eloss+=instant_eloss

            imse_list.append(instant_iloss)
            emse_list.append(instant_eloss)
            print('Item: {}/{}, iMSE: {}, eMSE: {}, cumul_iMSE: {}, cumul_eMSE: {}, time: {}'.format(j, len(self.dl['valid']),
                                                                                       instant_iloss, instant_eloss,
                                                                                       total_iloss/total_inum,
                                                                                       total_eloss/total_enum,
                                                                                       burden_t))
            recoreded_list = [('fourier_iMSE', instant_iloss), ('fourier_eMSE', instant_eloss),
                              ('cumul_iMSE', total_iloss/total_inum), ('cumul_eMSE', total_eloss/total_enum)]
            for l_id, loss in recoreded_list:
                draw_learning_curve(writer=self.writer, id=l_id, loss=loss, iter=j)

        imse_list = np.array(imse_list)
        emse_list = np.array(emse_list)
        self.record_result(imse=np.mean(imse_list), istd=np.std(imse_list),
                           emse=np.mean(emse_list), estd=np.std(emse_list),
                           avg_time=total_time/len(self.dl['valid']))

    def record_result(self, imse, istd, emse, estd, avg_time):
        path = "./results"
        self.check_direcory(path)
        df_file_name = "./results/{}.csv".format('1125_fourier')  # .format(args.exp_idx)
        date = time.strftime('%c', time.localtime(time.time()))
        df_res = pd.DataFrame({"model": [self.args.exid], "iMSE": [imse.item()], "istd":[istd], "eMSE": [emse.item()], "estd":[estd],"avg_time":[avg_time],"data":[date], "seed":[self.seed]})
        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)

    def check_direcory(self, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)  # {}".format(args.exp_idx)
