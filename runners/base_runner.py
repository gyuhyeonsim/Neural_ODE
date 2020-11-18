import torch
import random
from torch.utils.tensorboard import SummaryWriter

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

    def train(self):
        pass