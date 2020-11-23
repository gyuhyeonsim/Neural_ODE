import os

import torch
import numpy as np
import torch.nn as nn
import random

from torch.utils.tensorboard import SummaryWriter
from utils.run_utils.tensorboard_utils import *

from runners.base_runner import Runner

class LatentRunner(Runner):
    def __init__(self, args, dataloader, model, optim):
        super(LatentRunner, self).__init__(args, dataloader, model, optim)
        self.noise_std = .3

    def train(self):
        # train
        iter = 0
        best_mse = float('inf')
        self.model.train()
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                self.optimizer.zero_grad()
                t = b['t'][0].to(self.torch_device).float()[:self.args.dataset['interpolation']]  # all data entries share the same time point
                point = b['point'].to(self.torch_device).float()[:,:self.args.dataset['interpolation']]
                pred, loss = self.model(point, t)
                loss.backward()
                self.optimizer.step()
                print("[Train] epoch:{}/{}, iter:{}/{}, KL_div: {}".format(i,self.args.niter,
                                                                 j, len(self.dl['train']),
                                                                 loss.item()))
                draw_learning_curve(self.writer, id='t_loss', loss=loss.item(), iter=iter)
                iter+=1

            # interpolation loss & extrapolation loss
            i_loss, e_loss = self.valid(self.dl['valid'])
            print("[Validation] epoch:{}/{}, iter:{}/{}, i_mse: {}, e_mse: {}".format(i,self.args.niter,
                                                                          iter,len(self.dl['train'])*self.args.niter,
                                                                          i_loss.item(), e_loss.item()))

            for l_id, loss in [('valid_i_loss', i_loss.item()), ('valid_e_loss', e_loss.item())]:
                draw_learning_curve(writer=self.writer, id=l_id, loss=loss, iter=i)

            if best_mse>i_loss:
                best_mse = i_loss
                self.model_save(epoch=i, loss=i_loss)

    def valid(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            interpolation_num = 0
            extrapolation_num = 0
            total_interpolation_loss = 0
            total_extrapolation_loss = 0
            for i, b in enumerate(dataloader):
                t = b['t'][0].to(self.torch_device).float()  # all data entries share the same time point
                point = b['point'].to(self.torch_device).float()
                pred, loss = self.model(point, t)

                interpolation_loss = (pred-point)[:,:self.args.dataset['interpolation']]
                extrapolation_loss = (pred-point)[:,self.args.dataset['interpolation']:]
                interpolation_num += torch.ones_like(interpolation_loss).sum().item()
                extrapolation_num += torch.ones_like(interpolation_loss).sum().item()
                total_interpolation_loss += torch.pow(interpolation_loss,2).sum()
                total_extrapolation_loss += torch.pow(extrapolation_loss,2).sum()

        return total_interpolation_loss/interpolation_num, total_extrapolation_loss/extrapolation_num
