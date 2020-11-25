def get_runner(args, dataloader, model, optim):
    if args.model['name'] == 'vanilla' or args.model['name'] == 'generative':
        from runners.base_runner import Runner
        runner = Runner(args, dataloader, model, optim)

    elif args.model['name'] =='latentode':
        from runners.latent_runner import LatentRunner
        runner = LatentRunner(args, dataloader, model, optim)

    elif args.model['name'] =='fourier_series':
        from runners.no_optim_runner import NoOptimRunner
        runner = NoOptimRunner(args, dataloader, model, optim)

    return runner