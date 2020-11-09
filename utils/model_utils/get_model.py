from models.vanilla_node import NeuralODE
import torch

def get_model(args):
    model = NeuralODE(args)
    print("model: {}, number of params: {}".format(args.model['name'], count_parameters(model)))

    return model, get_optimizer(args, model)

def get_optimizer(args, model):
    if args.model['optimizer']=='ADAM':
        print('optimizer: {}'.format(args.model['optimizer']))
        optim = torch.optim.Adam(model.parameters(), args.lr, amsgrad=False)
    return optim

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_pretrained_dict(args):
    return torch.load(args.pretrained['path'])