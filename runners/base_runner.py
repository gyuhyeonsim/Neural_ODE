import os

import torch
import torch.nn as nn
import random

from torch.utils.tensorboard import SummaryWriter
from utils.run_utils.tensorboard_utils import *

class Runner():
    def __init__(self, args, data_loader, model, optimizer):
        self.args = args
        self.dl = data_loader
        self.torch_device = torch.device("cuda")
        self.model = model.to(self.torch_device)
        self.optimizer = optimizer
        self.seed = random.randint(0, 10000)
        self.writer = SummaryWriter("./runs/{}".format(self.args.exid))
        self.metric = dict()

        # for debug
        self.debug = args.debug  # True if debug mode
        self.cal_loss = nn.MSELoss()

    def train(self):
        # train
        iter = 0
        best_mse = float('inf')
        self.model.train()
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['valid']):
                self.optimizer.zero_grad()
                t = b['t'][0].to(self.torch_device).float()[:self.args.dataset['interpolation']]  # all data entries share the same time point
                point = b['point'].to(self.torch_device).float()[:,:self.args.dataset['interpolation']]
                freq = b['freq'].unsqueeze(1).float()
                amp = b['amp'].unsqueeze(1).float()
                sign = b['sign'].unsqueeze(1).float()
                latent_v = torch.cat([freq, amp, sign], dim=1).to(self.torch_device)
                #print(t.size(), point.size(), freq.size(), amp.size())
                # >> torch.Size([64, 100]) torch.Size([64, 100, 1]) torch.Size([64]) torch.Size([64])
                # print(torch.cat([freq, amp, sign], dim=1).size())
                # >> torch.Size([64, 3])
                pred = self.model(t, point[:,0,:], latent_v=latent_v).permute(1,0,2)
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

    def get_loss_mask(self, t):
        zero_v = torch.zeros_like(t)
        one_v = torch.ones_like(t)
        interpolation_mask = torch.cat([one_v[:self.args.dataset['interpolation']],zero_v[self.args.dataset['interpolation']:]],dim=0)
        extrapolation_mask = torch.cat([zero_v[:self.args.dataset['interpolation']],one_v[self.args.dataset['interpolation']:]],dim=0)
        return interpolation_mask, extrapolation_mask

    def record_loss(self, loss, epoch):
        pass