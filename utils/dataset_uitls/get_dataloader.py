def get_dataloader(args):
    if args.dataset['type'] == 'spiral':
        from datasets.spiral_dataloader import get_data_loader
        dataloader = get_data_loader(args)
        args.model['input_size'] = 2

    elif args.dataset['type'] == 'sine':
        from datasets.sinusoidal_dataloader import get_data_loader
        dataloader = get_data_loader(args)
        args.model['input_size'] = 1

    elif args.dataset['type'] == 'irregular':
        from datasets.irregular_sinusoidal_dataloader import get_data_loader
        dataloader = get_data_loader(args)
        args.model['input_size'] = 1

    elif args.dataset['type'] == 'synthetic':
        # for CHIL 2021
        if args.model['name'] == 'rmsn' and args.model['phase']==3\
                and args.dataset['truncate']:
            from datasets.rmsn_truncated_datasets import get_data_loader
            args.model['input_size'] = 2
            dataloader = get_data_loader(args)
        else:
            from datasets.synthetic import get_data_loader
            dataloader = get_data_loader(args)
            args.model['input_size'] = 2

    assert dataloader != None
    return dataloader