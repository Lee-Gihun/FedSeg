import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

__all__ = ["MODELS", "NUM_CLASSES", "OPTIMIZER", "SCHEDULER"]


NUM_CLASSES = {
    "acdc": 19,
}

MODELS = {
    "manet": smp.MAnet,
}

OPTIMIZER = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}
