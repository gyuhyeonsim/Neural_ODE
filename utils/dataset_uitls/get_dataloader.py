def get_dataloader(args):
    if args.dataset['type'] == 'spiral':
        from datasets.spiral_dataloader import get_data_loader
        dataloader = get_data_loader(args)
        args.model['input_size'] = 2

    assert dataloader != None
    return dataloader