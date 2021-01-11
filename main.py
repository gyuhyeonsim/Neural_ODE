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

if args.gpus is not None:
    # device = torch.device('cuda:' + str(args.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus   # It allows multi GPU settings
    args.device = torch.device("cuda")

tmp = args.exid
for i in range(1,4):
    args.model['phase'] = i
    args.exid = tmp+'_cv_'+str(args.cv_idx)
    print('Experiments {} is starts'.format(args.exid))
    dataloader = get_dataloader(args)
    model, optim = get_model(args)
    runner = get_runner(args, dataloader, model, optim)
    runner.train()
    print('success flag')