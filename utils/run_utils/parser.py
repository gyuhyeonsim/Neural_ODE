import argparse

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--config', type=str, default='configs/coeff_2layers_subtraction.yaml')

        # hyper parameters
        self.parser.add_argument('--num_hidden', type=int, default=64)
        self.parser.add_argument('--latent_dim', type=int, default=3)
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--niter', type=int, default=4000)

    def get_args(self):
        args = self.parser.parse_args()
        return args