from models.generative_ode import GalerkinDE
import torch

def get_model(args):
    if args.model['name'] == 'generative':
        model = GalerkinDE(args)
        optim, scheduler = get_optimizer(args, model)

    print("model: {}, number of params: {}".format(args.model['name'], count_parameters(model)))
    return model, optim, scheduler


def get_optimizer(args, model):
    if args.model['optimizer'] == 'ADAM':
        print('optimizer: {}'.format(args.model['optimizer']))
        optim = torch.optim.Adam(model.parameters(), args.lr, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5)
    return optim, scheduler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained_dict(args):
    return torch.load(args.pretrained['path'])