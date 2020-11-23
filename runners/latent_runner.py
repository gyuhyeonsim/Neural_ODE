import os

import torch
import torch.nn as nn
import random

from torch.utils.tensorboard import SummaryWriter
from utils.run_utils.tensorboard_utils import *

from runners.base_runner import Runner

class LatentRunner(Runner):
    def __init__(self, args, dataloader, model, optim):
        super(LatentRunner, self).__init__(args, dataloader, model, optim)

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
                # freq = b['freq'].unsqueeze(1).float()
                # amp = b['amp'].unsqueeze(1).float()
                # sign = b['sign'].unsqueeze(1).float()
                # latent_v = torch.cat([freq, amp, sign], dim=1).to(self.torch_device)
                pred = self.model(point, t)
                loss = self.cal_loss(pred, point)
                loss.backward()
                self.optimizer.step()

                print("[Train] epoch:{}/{}, iter:{}/{}, loss: {}".format(i,self.args.niter,
                                                                 j, len(self.dl['train']),
                                                                 loss.item()))
                draw_learning_curve(self.writer, id='t_loss', loss=loss.item(), iter=iter)
                iter+=1
            # interpolation loss & extrapolation loss
            i_loss, e_loss = self.valid(self.dl['valid'])
            print("[Validation] epoch:{}/{}, iter:{}/{}, i_loss: {}, e_loss: {}".format(i,self.args.niter,
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
            valid_loss = 0
            interpolation_num = 0
            extrapolation_num = 0
            total_interpolation_loss = 0
            total_extrapolation_loss = 0
            for i, b in enumerate(dataloader):
                t = b['t'][0].to(self.torch_device).float()  # all data entries share the same time point
                point = b['point'].to(self.torch_device).float()
                freq = b['freq'].unsqueeze(1).float()
                amp = b['amp'].unsqueeze(1).float()
                sign = b['sign'].unsqueeze(1).float()
                latent_v = torch.cat([freq, amp, sign], dim=1).to(self.torch_device)
                pred = self.model(t, point[:,0,:], latent_v=latent_v).permute(1,0,2)
                self.get_loss_mask(t)

                interpolation_loss = (pred-point)[:,:self.args.dataset['interpolation']]
                extrapolation_loss = (pred-point)[:,self.args.dataset['interpolation']:]
                interpolation_num += torch.ones_like(interpolation_loss).sum().item()
                extrapolation_num += torch.ones_like(interpolation_loss).sum().item()
                total_interpolation_loss += torch.pow(interpolation_loss,2).sum()
                total_extrapolation_loss += torch.pow(extrapolation_loss,2).sum()

        return total_interpolation_loss/interpolation_num, total_extrapolation_loss/extrapolation_num

    def cal_loss(self, pred, gt):
        pass