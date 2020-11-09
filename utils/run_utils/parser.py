import argparse

class Parser:
    def __init__(self):
        # general purpose
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data', type=str, choices=['simple_siusoid', '3d_cell'], default='simple_siusoid')
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--cpus', type=int, default=32) # In case of 3d cell, it is 32 default.
        self.parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
        self.parser.add_argument('--exid', type=str, default='default')
        self.parser.add_argument('--model', type=str, choices=['2d_unet', '3d_cell'], default='2d_unet')
        self.parser.add_argument('--config', type=str)
        self.parser.add_argument('--test_seed', type=str, default=None)

        # hyper parameters
        self.parser.add_argument('--num_hidden', type=int, default=64)
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--niter', type=int, default=200)

    def get_args(self):
        args = self.parser.parse_args()
        if args.data =='sinusoid':
            args.path = '/data02/jinho/data/lung_ct/'
        return args