
def get_dataloader(args):
    if args.dataset_name == 'paired_3':
        from .dataloader_paired_3 import get_dataloader as f
    elif args.dataset_name == 'paired':
        from .dataloader_paired import get_dataloader as f
    elif args.dataset_name == 'paired_4':
        from .dataloader_paired_4 import get_dataloader as f
    else:
        raise NotImplementedError("{} is not supported".format(args.dataset_name))
    
    return f(args)
    
def get_eval_dataloader(args):
    if args.dataset_name == 'paired_3':
        from .dataloader_paired_3 import get_eval_dataloader as f
    elif args.dataset_name == 'paired':
        from .dataloader_paired import get_eval_dataloader as f
    elif args.dataset_name == 'paired_4':
        from .dataloader_paired_4 import get_eval_dataloader as f
    else:
        raise NotImplementedError("{} is not supported".format(args.dataset_name))
    
    return f(args)
    
    