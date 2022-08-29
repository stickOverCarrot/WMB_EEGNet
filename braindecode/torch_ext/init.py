from torch.nn import init
import torch


def glorot_weight_zero_bias(model):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.
    
    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        # if isinstance(module, torch.nn.ModuleList):
        #     glorot_weight_zero_bias(module)
        # elif isinstance(module, torch.nn.Sequential):
        #     glorot_weight_zero_bias(module)
        # else:
        if hasattr(module, "weight"):
            if "GroupNorm" in module.__class__.__name__:
                continue
            if not ("BatchNorm" in module.__class__.__name__):
                init.xavier_uniform_(module.weight, gain=1)
            else:
                init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None and not isinstance(module.bias, bool):
                init.constant_(module.bias, 0)
