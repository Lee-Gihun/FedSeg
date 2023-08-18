import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import argparse
import warnings
import wandb
import random
import pprint
import os

import algorithms
from train_tools import *
from SetupDict import NUM_CLASSES, MODELS, OPTIMIZER, SCHEDULER


warnings.filterwarnings("ignore")

# Set torch base print precision
torch.set_printoptions(10)

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "fedavgm": algorithms.fedavgm.Server,
    "fedprox": algorithms.fedprox.Server,
    "fedlab": algorithms.fedlab.Server,
    "fedntd": algorithms.fedntd.Server,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}


def _get_setups(args):
    """Get train configuration"""

    # # Distribute the data to clients
    data_distributed = data_distributer(**args.data_setups)

    # Fix randomness for experiment
    fix_randomness(args.train_setups.seed)

    # Create a federated model
    num_classes = NUM_CLASSES[args.data_setups.dataset_name]
    args.model_setups.params.classes = num_classes
    model = MODELS[args.model_setups.name](**args.model_setups.params)

    # Optimization setups
    optimizer = OPTIMIZER[args.train_setups.optimizer.name](
        model.parameters(), **args.train_setups.optimizer.params
    )
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    # Algorith-specific global server container
    algo_params = args.train_setups.algo.params
    server = None
    server = ALGO[args.train_setups.algo.name](
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        **args.train_setups.scenario,
    )

    # if args.train_setups.pretrained.enabled:
    #     pretrained = torch.load(
    #         args.train_setups.pretrained.path, map_location=torch.device("cpu")
    #     )
    #     server.server_model.load_state_dict(pretrained)
    #     print("\nPretrained model loaded...")

    return server


def main(args):
    """Execute experiment"""

    # Load the configuration
    server = _get_setups(args)

    # # Conduct FL
    server.run()

    # Save the final global model
    # model_path = os.path.join(wandb.run.dir, f"{args.train_setups.algo.name}.pth")
    # torch.save(server.server_model.state_dict(), model_path)

    # Upload model to wandb
    # wandb.save(model_path)


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--config_path", default="./config/fedavg.json", type=str)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Overwrite config by parsed arguments
    # opt = config_overwriter(opt, args)

    # Print configuration dictionary pretty
    print("")
    print("\n" + "=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    # Initialize W&B
    wandb.init(config=opt, **opt.wandb_setups)

    # Execute expreiment
    main(opt)
