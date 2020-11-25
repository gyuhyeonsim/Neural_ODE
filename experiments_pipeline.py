import torch
import torch.nn as nn
import yaml
import os

from utils import Parser, get_dataloader, get_model, get_runner

args = Parser().get_args()
if args.config is not None:
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        args.__setattr__(k, v)

for i in range(1, 11):
    args.model['n'] = i
    args.exid = 'fourier_series_'+str(i)
    print("[System] {} th experiment starts, exid:{}".format(args.model['n'], args.exid))

    dataloader = get_dataloader(args)
    model, optim = get_model(args)

    runner = get_runner(args, dataloader, model, optim)
    runner.train()

    print('success flag')