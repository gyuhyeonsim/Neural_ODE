def get_runner(args, dataloader, model, optim=None):
    if args.model['name'] == 'vanilla' or args.model['name'] == 'generative':
        from runners.base_runner import Runner
        runner = Runner(args, dataloader, model, optim)

    elif args.model['name'] =='latentode':
        from runners.latent_runner import LatentRunner
        runner = LatentRunner(args, dataloader, model, optim)

    elif args.model['name'] =='fourier_series':
        from runners.no_optim_runner import NoOptimRunner
        runner = NoOptimRunner(args, dataloader, model, optim)

    elif args.model['name'] =='irregular' or args.model['name'] =='siren':
        from runners.irregular_runner import IrregularRunner
        runner = IrregularRunner(args, dataloader, model, optim)

    elif args.model['name'] == 'rmsn':
        if args.model['phase'] == 1:
            from runners.propensity_runner import PropensityRunner
            runner = PropensityRunner(args, dataloader, model)
    return runner