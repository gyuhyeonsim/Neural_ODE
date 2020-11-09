import torch
import torch.nn as nn
import yaml
import os

from utils import Parser, get_dataloader, get_model
from runners.base_runner import Runner

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

dataloader = get_dataloader(args)
model, optim = get_model(args)
runner = Runner(args, dataloader, model, optim)
runner.train()

print('success flag')