# from .old.SRN_model import SRN_Net
# from .old.SRNATT_model import SRNATT_Net
# from .old.SRNATTS_model import SRNATTS_Net
# from .old.UNet_model import U_Net
# from .ATT_model import ATT_Net
# from .ATT_Deblur_model import ATT_Deblur_Net
# from .old.ATT_Deblur_model_level1 import ATT_Deblur_Net_level1
# from .old.ATT_Deblur_model_level2 import ATT_Deblur_Net_level2
# from .old.ATT_Deblur_model_level3 import ATT_Deblur_Net_level3
# from .old.ATT_Deblur_model_level4 import ATT_Deblur_Net_level4

# def get_model(args):
#     if args.model == 'U_Net':
#         return U_Net(args)
#     elif args.model == 'SRN_Net':
#         return SRN_Net(args)
#     elif args.model == 'SRNATT_Net':
#         return SRNATT_Net(args)
#     elif args.model == 'SRNATTS_Net':
#         return SRNATTS_Net(args)
#     elif args.model == 'ATT_Net':
#         return ATT_Net(args)
#     elif args.model == 'ATT_Deblur_Net':
#         return ATT_Deblur_Net(args)
#     elif args.model == 'ATT_Deblur_Net_level1':
#         return ATT_Deblur_Net_level1(args)
#     elif args.model == 'ATT_Deblur_Net_level2':
#         return ATT_Deblur_Net_level2(args)
#     elif args.model == 'ATT_Deblur_Net_level3':
#         return ATT_Deblur_Net_level3(args)
#     elif args.model == 'ATT_Deblur_Net_level4':
#         return ATT_Deblur_Net_level4(args)
#     else:
#         raise NotImplementedError("{} is not supported".format(args.model))
    
from .ATT_Deblur_model_all_level import ATT_Deblur_Net
from .ATT_Deblur_trans import ATT_Trans_Deblur_Net

def get_model(args):
    if args.model == 'ATT_Trans_Deblur_Net':
        return ATT_Trans_Deblur_Net(args)
    elif args.model == "ATT_Deblur_Net":
        return ATT_Deblur_Net(args)
    else:
        raise NotImplementedError("{} is not supported".format(args.model))
