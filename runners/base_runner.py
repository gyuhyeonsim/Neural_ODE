import os
import logging
import torch
import torch.nn as nn
import random

from utils.run_utils.runner_utils import log

class Runner():
    def __init__(self, args, train_data_loader, valid_data_loader, model, optimizer, scheduler):
        self.args = args
        self.dl = train_data_loader
        self.val_dl = valid_data_loader
        self.torch_device = torch.device('cuda')
        self.model = model.to(self.torch_device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.filename = self.args.model['filename']
        self.logger = log(path=self.args.dataset['output_dir'], file= self.filename + '.logs')
        self.path = self.args.dataset['output_dir'] + self.filename + '.pt'

    def train(self):
        best_mse = float('inf')
        if os.path.exists(self.path):
            checkpoint = torch.load(self.path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_mse = checkpoint['loss']

        self.model.train()
        self.logger.info('dataset type: {}'.format(str(self.args.dataset['type'])))
        self.logger.info('filename: {}'.format(str(self.filename)))
        self.logger.info('description: {}'.format(str(str(self.args.description))))
        for i in range(self.args.niter):
            for j, sample in enumerate(self.dl):
                self.optimizer.zero_grad()
                samp_sin, samp_ts, latent_v = sample
                samp_sin = samp_sin.to(self.args.device)
                samp_ts = samp_ts.to(self.args.device)
                latent_v = latent_v.to(self.args.device)

                train_loss, dilation_penalty = self.model(samp_ts, samp_sin, latent_v)

                train_loss.backward()
                self.optimizer.step()

            if i % 20 == 0:
                self.logger.info("[Train] epoch:{}/{}, iter:{}/{}, loss: {}, dilation_penalty: {}".format(i,self.args.niter,
                                                                 j, len(self.dl),
                                                                 train_loss.item(),
                                                                 dilation_penalty.item()))

            valid_loss, valid_dilation_penalty = self.valid(self.val_dl)
            self.scheduler.step(valid_loss)

            if i % 20 == 0:
                self.logger.info("[Valid] epoch:{}/{}, loss: {}, dilation_penalty: {}".format(i,self.args.niter, valid_loss.item(), valid_dilation_penalty.item()))

            if best_mse > valid_loss:
                best_mse = valid_loss
                #self.model_save(epoch = i, loss=valid_loss)
                self.logger.info("[Valid] epoch:{}/{}, loss: {}".format(i, self.args.niter, valid_loss.item()))

    def valid(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            valid_loss = 0
            for i, sample in enumerate(dataloader):
                samp_sin, samp_ts, latent_v = sample
                samp_sin = samp_sin.to(self.args.device)
                samp_ts = samp_ts.to(self.args.device)
                latent_v = latent_v.to(self.args.device)

                loss, dilation_penalty = self.model(samp_ts, samp_sin, latent_v)
                valid_loss += loss
        return valid_loss, dilation_penalty

    def model_save(self, epoch, loss):
        print("saved epoch {}".format(epoch))
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss}, self.path)












