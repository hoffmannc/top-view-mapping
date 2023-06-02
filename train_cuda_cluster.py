import yaml
import wandb

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from src.data.dataset import NuScencesMaps
from src.model.network import PyrOccTranDetr_S_0904_old_2DPosEncBEV as Model
from src.utils import dice_loss_mean
from src.trainer import Trainer


def main():
    # Mulit-GPU
    init_process_group(backend="nccl")

    # Configuration
    with open("conf/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Data
    train_data = NuScencesMaps(
        config["paths"]["nuscenes"], config["training"]["train_split"]
    )
    trainloader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
        sampler=DistributedSampler(train_data),
    )

    val_data = NuScencesMaps(
        config["paths"]["nuscenes"], config["training"]["val_split"]
    )
    valloader = DataLoader(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
        sampler=DistributedSampler(val_data),
    )

    # Model
    model = Model(**config["model"])

    # Loss
    criterion = dice_loss_mean

    # Optimizer
    optimizer = optim.Adam(model.parameters(), config["training"]["lr"])

    # Reproducability
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # WandB
    wandb.init(
        project="top-view-mapping",
        entity="hoffmann",
        name="Top View Mapping (LSF-10)",
        config=config,
        group="DDP",
    )

    # MAIN
    trainer = Trainer(model, trainloader, valloader, optimizer, criterion, config)
    torch.cuda.empty_cache()
    trainer.train()

    destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    main()
