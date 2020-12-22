import os
import torch
import torch.nn as nn
import random

from torch.utils.tensorboard import SummaryWriter
from utils.run_utils.tensorboard_utils import *

class PropensityRunner():
    def __init__(self, args, dataloader, model):
        self.args = args
        self.dl = dataloader
        self.model = model

    def train(self):
        # train
        iter = 0
        best_mse = float('inf')
        self.model.train()
        self.args.niter = 0
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                x = b['obs'].float()
                itv = b['itv'].float()
                t = b['t'].float()
                #torch.Size([64, 50, 2]) torch.Size([64, 50, 3]) torch.Size([50])
                # print(x.size(), itv.size(), t.size())
                self.model.st_n_optim.zero_grad()
                kld_loss, nll_loss, _, _ = self.model.st_numer(itv[:,:,:2].permute(1,0,2))
                loss = (kld_loss + nll_loss)/(itv.size(0)*itv.size(1)*2)
                loss.backward()
                self.model.st_n_optim.step()
                print('[Stabilized Weights Nominator] Epoch: {}/{}, Loss: {}'
                      .format(self.args.niter, i, round(loss.item(),4)))

        self.args.niter = 2
        # Train Stabilized Weights Denominator Network
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                x = b['obs'].float()
                itv = b['itv'].float()

                self.model.st_d_optim.zero_grad()
                H=torch.cat([x[:,1:,:], itv[:,:itv.size(1)-1,:2]],dim=2).permute(1,0,2)
                kld_loss, nll_loss, _, _ = self.model.st_denor(x=H, gt=itv[:, :, :2].permute(1, 0, 2))
                loss = (kld_loss + nll_loss) / (itv.size(0) * itv.size(1) * 2)
                loss.backward()
                self.model.st_d_optim.step()
                print('[Stabilized Weights Denominator] Epoch:{}/{}, Loss: {}'
                      .format(self.args.niter, i, round(loss.item(),4)))

        self.args.niter=0
        # Train Censoring Nominator Network
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                x = b['obs'].float()
                itv = b['itv'].float()
                self.model.cs_n_optim.zero_grad()
                loss = self.model.censor_numer(itv[:,:,:2].permute(1,0,2), gt=itv[:,:,2:3].permute(1,0,2))
                loss.backward()
                self.model.cs_n_optim.step()
                print('[Stabilized Weights Nominator] Epoch: {}/{}, Loss: {}'
                      .format(self.args.niter, i, round(loss.item(),4)))

        self.args.niter=10
        # Train Censoring Denominator Network
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                x = b['obs'].float()
                itv = b['itv'].float()
                self.model.cs_d_optim.zero_grad()
                H=torch.cat([x, itv[:,:,:2]],dim=2).permute(1,0,2)
                loss = self.model.censor_denor(x=H, gt=itv[:,:,2:3].permute(1,0,2))
                loss.backward()
                self.model.cs_d_optim.step()
                print('[Stabilized Weights Nominator] Epoch: {}/{}, Loss: {}'
                      .format(self.args.niter, i, round(loss.item(),4)))

    def valid(self, dataloader):
        pass

    def model_save(self, epoch, loss):
        print("saved epoch {}".format(epoch))
        path = "./save/{}".format(self.args.exid)
        self.check_direcory('./save')
        self.check_direcory(path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        },path+"/best_model_"+str(self.seed))

    def check_direcory(self, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)  # {}".format(args.exp_idx)