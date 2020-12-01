import torch
import torch.nn as nn
import yaml
import os

from utils.run_utils.parser import Parser
from utils.model_utils.get_model import get_model
from datasets.sinusoidal_dataloader import get_dataloader
from runners.base_runner import Runner

args = Parser().get_args()
if args.config is not None:
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        args.__setattr__(k, v)

if args.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device("cuda")

args.description = 'dataset1, dilation only'

train_dataloader = get_dataloader(args)
valid_dataloader = get_dataloader(args)
model, optim, scheduler = get_model(args)

runner = Runner(args=args, train_data_loader=train_dataloader, valid_data_loader=valid_dataloader, \
                model=model, optimizer=optim, scheduler=scheduler)

runner.train()

print('success flag')


