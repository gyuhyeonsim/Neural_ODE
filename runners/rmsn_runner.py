import os
import torch
import torch.nn as nn
import random

from models.rmsn.encoder_net import EncoderNet
from models.rmsn.decoder_net import DecoderNet
from torch.utils.tensorboard import SummaryWriter
from utils.run_utils.tensorboard_utils import *

class PropensityRunner():
    def __init__(self, args, dataloader, model):
        self.args = args
        self.dl = dataloader
        self.model = model.to(args.device)
        self.writer = SummaryWriter("./rmsn/{}".format(self.args.exid))

    def train(self):
        # train
        if self.args.model['phase']==1:
            self.train_propensity()

        elif self.args.model['phase']==2:
            ## add model
            self.encoder = self.get_encoder()
            self.encoder_optim =torch.optim.Adam(self.encoder.parameters(),
                                                 self.args.model['encoder_lr'],
                                                 amsgrad=False)
            self.train_encoder()

        elif self.args.model['phase'] == 3:
            # self.encoder = self.get_encoder()
            if self.args.dataset['truncate']:
                self.model = None  # remove the reference to PropensityNet
                self.decoder = self.get_decoder()
                self.decoder_optim = torch.optim.Adam(self.decoder.parameters(),
                                                      self.args.model['decoder_lr'], amsgrad=False)
                self.train_decoder()
            else:
                self.encoder = self.get_encoder()
                self.decoder = self.get_decoder()
                self.decoder_optim = torch.optim.Adam(self.decoder.parameters(),
                                                      self.args.model['decoder_lr'], amsgrad=False)
                self.train_non_truncated_decoder()

        elif self.args.model['phase'] == 'data_gen':
            # generation a truncated datasets
            self.encoder = self.get_encoder()

    def train_propensity(self):
        self.args.niter = 0
        best_mse = float('inf')
        self.model.train()
        self.args.niter = 300
        total_iter=0
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                itv = b['itv'].float().to(self.args.device)
                #torch.Size([64, 50, 2]) torch.Size([64, 50, 3]) torch.Size([50])
                # print(x.size(), itv.size(), t.size())
                self.model.st_n_optim.zero_grad()
                # print(itv[:,:,:2].permute(1,0,2).size()) [50, 64, 2]
                # [:,:,:2] means slicing mask
                mse_loss,_ = self.model.st_numer(
                    itv[:,:,:self.args.model['intervention_dim']].permute(1,0,2))
                loss = (mse_loss)/(itv.size(0)*itv.size(1)*self.args.model['intervention_dim'])
                loss.backward()
                self.model.st_n_optim.step()

                print('[Stabilized Weights Nominator] Epoch: {}/{}, Loss: {}'
                      .format(i, self.args.niter, round(loss.item(),4)))
                self.draw_learning_curve(self.writer,'train_st_numer',loss.item(),total_iter)
                total_iter+=1

            valid_loss = 0
            valid_num = 0
            for j, b in enumerate(self.dl['valid']):
                itv = b['itv'].float().to(self.args.device)
                mse_loss,_ = self.model.st_numer(itv[:,:,:self.args.model['intervention_dim']].permute(1,0,2))
                valid_loss += mse_loss
                valid_num += itv.size(0)*itv.size(1)*self.args.model['intervention_dim']
            valid_loss /= valid_num

            print('[Stabilized Weights Nominator] Epoch: {}/{}, Valid Loss: {}'
                  .format(i, self.args.niter, round(valid_loss.item(),4)))
            self.draw_learning_curve(self.writer, 'valid_st_numer', valid_loss.item(), i)

            if best_mse>valid_loss:
                best_mse = valid_loss
                self.model_save(epoch=i, loss=best_mse, name='SW_numer',model=self.model.st_numer)
                print('[Stabilized Weights Nominator] Saved at {}'.format(i))

        best_mse = float('inf')
        total_iter = 0
        # Train Stabilized Weights Denominator Network
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                x = b['obs'].float().to(self.args.device)
                itv = b['itv'].float().to(self.args.device)
                self.model.st_d_optim.zero_grad()
                H = torch.cat([x[:,1:,:], itv[:,:itv.size(1)-1,:self.args.model['intervention_dim']]],dim=2).permute(1,0,2)
                mse_loss, _ = self.model.st_denom(x=H, gt=itv[:,1:,:self.args.model['intervention_dim']].permute(1, 0, 2))
                loss = (mse_loss) / (itv.size(0) * itv.size(1) * 1) # censoring variable
                loss.backward()
                self.model.st_d_optim.step()
                print('[Stabilized Weights Denominator] Epoch:{}/{}, Loss: {}'
                      .format(i, self.args.niter, round(loss.item(),4)))
                self.draw_learning_curve(self.writer,'train_st_denom',loss.item(),total_iter)
                total_iter+=1

            valid_loss = 0
            valid_num = 0
            for j, b in enumerate(self.dl['valid']):
                x = b['obs'].float().to(self.args.device)
                itv = b['itv'].float().to(self.args.device)
                H=torch.cat([x[:,1:,:], itv[:,:itv.size(1)-1,:self.args.model['intervention_dim']]],dim=2).permute(1,0,2)
                mse_loss, _ = self.model.st_denom(x=H, gt=itv[:,1:,:self.args.model['intervention_dim']].permute(1, 0, 2))
                valid_loss += mse_loss
                valid_num += (itv.size(0) * itv.size(1) * 1) #self.args.model['intervention_dim']
            valid_loss/=valid_num

            print('[Stabilized Weights Denominator] Epoch: {}/{}, Valid Loss: {}'
                  .format(i, self.args.niter, round(valid_loss.item(),4)))
            self.draw_learning_curve(self.writer, 'valid_st_denom', valid_loss.item(), i)

            if best_mse>valid_loss:
                best_mse = valid_loss
                self.model_save(epoch=i, loss=best_mse, name='SW_deno',model=self.model.st_denom)
                print('[Stabilized Weights Denominator] Saved at {}'.format(i))

        best_mse = float('inf')
        total_iter = 0
        # Train Censoring Nominator Network
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                itv = b['itv'].float().to(self.args.device)
                self.model.cs_n_optim.zero_grad()
                # gt is mask of intervention
                loss = self.model.censor_numer(itv[:,:,:self.args.model['intervention_dim']].permute(1,0,2),
                                               gt=itv[:,:,self.args.model['intervention_dim']:].permute(1,0,2))
                loss /= torch.ones_like(itv[:,:,self.args.model['intervention_dim']:]).sum()
                loss.backward()
                self.model.cs_n_optim.step()
                print('[Censoring Nominator] Epoch: {}/{}, Loss: {}'
                      .format(i, self.args.niter, round(loss.item(),4)))
                self.draw_learning_curve(self.writer,'train_cens_numer',loss.item(),total_iter)
                total_iter+=1

            valid_loss = 0
            valid_num = 0
            for j, b in enumerate(self.dl['valid']):
                itv = b['itv'].float().to(self.args.device)
                loss = self.model.censor_numer(itv[:,:,:self.args.model['intervention_dim']].permute(1,0,2),
                                               gt=itv[:,:,self.args.model['intervention_dim']:].permute(1,0,2))
                valid_loss += loss
                valid_num += torch.ones_like(itv[:,:,self.args.model['intervention_dim']:]).sum()

            valid_loss/=valid_num
            print('[Censoring Nominator] Epoch: {}/{}, Valid Loss: {}'
                  .format(i, self.args.niter, round(valid_loss.item(),4)))
            self.draw_learning_curve(self.writer, 'valid_cens_numer', valid_loss.item(),i)

            if best_mse>valid_loss:
                best_mse = valid_loss
                self.model_save(epoch=i, loss=best_mse, name='CS_numer',model=self.model.censor_numer)
                print('[Censoring Nominator] Saved at {}'.format(i))

        best_mse = float('inf')
        total_iter=0

        # Train Censoring Denominator Network
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                x = b['obs'].float().to(self.args.device)
                itv = b['itv'].float().to(self.args.device)
                self.model.cs_d_optim.zero_grad()
                H=torch.cat([x, itv[:,:,:self.args.model['intervention_dim']]],dim=2).permute(1,0,2)
                loss = self.model.censor_denom(x=H, gt=itv[:,:,2:3].permute(1,0,2))
                loss/= torch.ones_like(itv[:,:,self.args.model['intervention_dim']:]).sum()
                loss.backward()
                self.model.cs_d_optim.step()
                print('[Censoring Denominator] Epoch: {}/{}, Loss: {}'
                      .format(i, self.args.niter, round(loss.item(),4)))
                self.draw_learning_curve(self.writer,'train_cens_denom',loss.item(),total_iter)
                total_iter+=1

            valid_loss = 0
            valid_num = 0
            for j, b in enumerate(self.dl['valid']):
                x = b['obs'].float().to(self.args.device)
                itv = b['itv'].float().to(self.args.device)
                H=torch.cat([x, itv[:,:,:self.args.model['intervention_dim']]],dim=2).permute(1,0,2)
                loss = self.model.censor_denom(x=H, gt=itv[:,:,self.args.model['intervention_dim']:].permute(1,0,2))
                valid_loss += loss
                valid_num += torch.ones_like(itv[:,:,self.args.model['intervention_dim']:]).sum()

            valid_loss/=valid_num
            print('[Censoring Denominator] Epoch: {}/{}, Valid Loss: {}'
                  .format(i, self.args.niter, round(valid_loss.item(),4)))

            self.draw_learning_curve(self.writer,'valid_cens_denom',valid_loss.item(),i)

            if best_mse>valid_loss:
                best_mse = valid_loss
                self.model_save(epoch=i, loss=best_mse, name='CS_denom',model=self.model.censor_denom)
                print('[Censoring Denominator] Saved at {}'.format(i))

    def train_encoder(self):
        best_mse = float('inf')
        self.model.train()
        # self.args.niter = 5
        total_iter = 0
        niter = self.args.model['encoder_epoch']
        for i in range(niter):
            for j, b in enumerate(self.dl['train']):
                self.encoder_optim.zero_grad()
                x = b['obs'].float()
                itv = b['itv'].float()
                state = torch.cat([x,itv[:,:,:2]],dim=2).permute(1,0,2)
                loss,_,_ = self.encoder(x=state, gt=x.permute(1,0,2))
                loss /= torch.ones_like(x).sum()
                loss.backward()
                self.encoder_optim.step()
                print('[Encoder] Epoch:{}/{}, Loss: {}'
                      .format(i, niter, round(loss.item(),4)))
                self.draw_learning_curve(self.writer,'train_encoder',loss.item(),total_iter)
                total_iter+=1

            valid_loss = 0
            valid_num = 0
            for j, b in enumerate(self.dl['valid']):
                x = b['obs'].float()
                itv = b['itv'].float()
                state = torch.cat([x, itv[:,:,:2]], dim=2).permute(1, 0, 2)
                valid_loss_add,_,_ = self.encoder(x=state, gt=x.permute(1, 0, 2), valid=True)
                valid_loss += valid_loss_add
                valid_num += torch.ones_like(x).sum()

            valid_loss/=valid_num
            self.draw_learning_curve(self.writer,'valid_encoder',valid_loss.item(),i)
            print('[Encoder] Epoch:{}/{}, Valid Loss: {}'
                  .format(i, niter, round(valid_loss.item(), 8)))
            if best_mse>valid_loss:
                best_mse = valid_loss
                self.model_save(epoch=i, loss=best_mse, name='Encoder',model=self.encoder)
                print('[Encoder] Saved at {}'.format(i))

    def train_decoder(self):
        best_mse = float('inf')
        self.decoder.train()
        # self.args.niter = 5
        total_iter = 0
        niter = self.args.model['decoder_epoch']
        for i in range(niter):
            for j, b in enumerate(self.dl['train']):
                self.decoder_optim.zero_grad()
                x = b['obs'].float()
                itv = b['itv'].float()
                mask = b['mask'].float()
                init_hidden = b['hidden'].float()
                weight = b['weight'].float()
                loss = self.decoder(init_hidden, itv[:,:,:-1], mask, weight, x)
                loss /= mask.sum()
                loss.backward()
                self.decoder_optim.step()
                print('[Decoder] Epoch:{}/{}, Loss: {}'
                      .format(i, niter, round(loss.item(), 6)))
                self.draw_learning_curve(self.writer, 'train_decoder', loss.item(), total_iter)
                total_iter += 1

            valid_loss = 0
            valid_num = 0
            for j, b in enumerate(self.dl['valid']):
                self.decoder_optim.zero_grad()
                x = b['obs'].float()
                itv = b['itv'].float()
                mask = b['mask'].float()
                init_hidden = b['hidden'].float()
                weight = b['weight'].float()
                valid_loss_add = self.decoder(init_hidden, itv[:,:,:-1], mask, weight, x, True)
                valid_loss += valid_loss_add
                valid_num += mask.sum()
            valid_loss /= valid_num
            self.draw_learning_curve(self.writer, 'valid_decoder', valid_loss.item(), i)

            print('[Decoder] Epoch:{}/{}, Valid Loss: {}'
                  .format(i, niter, round(valid_loss.item(), 8)))
            if best_mse > valid_loss:
                best_mse = valid_loss
                self.model_save(epoch=i, loss=best_mse, name='Decoder', model=self.decoder)
                print('[Encoder] Saved at {}'.format(i))
        print(best_mse)

    def train_non_truncated_decoder(self):
        state_dict = torch.load('./save/{}/Encoder'.format(self.args.exid))['model_state_dict']
        self.encoder.load_state_dict(state_dict)
        print('Succeed in Loading Encoder')
        best_mse = float('inf')
        self.decoder.train()

        total_iter = 0
        niter = self.args.model['decoder_epoch']
        for i in range(niter):
            for j, b in enumerate(self.dl['train']):
                self.decoder_optim.zero_grad()
                x = b['obs'].float()
                itv = b['itv'].float()
                loss = self.decoder.forward_nontruncated(x, itv, self.encoder)
                loss /= torch.ones_like(x[6:]).sum()
                # print(torch.ones_like(x[6:]).sum())
                loss.backward()
                self.decoder_optim.step()
                print('[Decoder(non-truncated)] Epoch:{}/{}, Loss: {}'
                      .format(i, niter, round(loss.item(), 6)))
                self.draw_learning_curve(self.writer, 'train_decoder', loss.item(), total_iter)
                total_iter += 1

            valid_loss = 0
            valid_num = 0
            for j, b in enumerate(self.dl['valid']):
                self.decoder_optim.zero_grad()
                x = b['obs'].float()
                itv = b['itv'].float()
                valid_loss_add = self.decoder.forward_nontruncated(x, itv, self.encoder, True)
                valid_loss += valid_loss_add
                valid_num += torch.ones_like(x[6:]).sum()
            valid_loss /= valid_num
            self.draw_learning_curve(self.writer, 'valid_decoder', valid_loss.item(), i)

            print('[Decoder(non-truncated)] Epoch:{}/{}, Valid Loss: {}'
                  .format(i, niter, round(valid_loss.item(), 8)))
            if best_mse > valid_loss:
                best_mse = valid_loss
                self.model_save(epoch=i, loss=best_mse, name='Decoder', model=self.decoder)
                print('[Decoder(non-truncated)] Saved at {}'.format(i))
        print(best_mse)

    def get_encoder(self):
        input_dim = self.args.model['intervention_dim'] + \
                    self.args.model['observation_dim']
        hidden_dim = self.args.model['encoder_dim']
        obs_dim = self.args.model['observation_dim']
        encoder = EncoderNet(input_dim, hidden_dim,
                                  obs_dim, propensity_net=self.model)
        return encoder

    def get_decoder(self):
        input_hidden_dim = self.args.model['encoder_dim']
        hidden_dim = self.args.model['decoder_hidden_dim']
        obs_dim = self.args.model['observation_dim']
        decoder = DecoderNet(input_hidden_dim, hidden_dim,
                             itv_dim=self.args.model['intervention_dim'],
                             obs_dim=obs_dim)
        return decoder

    def model_save(self, epoch, loss, name, model):
        # print("saved epoch {}".format(epoch))
        path = "./save/{}".format(self.args.exid)
        self.check_direcory('./save')
        self.check_direcory(path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        },path+"/"+name)

    def check_direcory(self, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)  # {}".format(args.exp_idx)

    def draw_learning_curve(self, writer, id, loss, iter):
        writer.add_scalar('./loss/' + id, loss, iter)

