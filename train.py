import yaml
import os
import sys

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from src.data.dataset_nuscenes import NuScencesMaps
from src.model.network import PyrOccTranDetr_S_0904_old_2DPosEncBEV as Model
from src.utils import dice_loss_mean
from src.trainer import Trainer


def main(config_name):
    # DDP
    init_process_group(backend="nccl")
    torch.cuda.empty_cache()

    # Configuration
    with open(f"conf/{config_name}.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Files
    gpu_id = int(os.environ["RANK"])
    if gpu_id == 0:
        # Logs
        filename = config["filename"]
        if not os.path.exists(f"log/{filename}"):
            os.mkdir(f"log/{filename}")
            open(f"log/{filename}/train_loss.txt", "w").close()
            open(f"log/{filename}/val_loss.txt", "w").close()

        # Checkpoints
        if not os.path.exists(os.path.join(config["paths"]["checkpoints"], filename)):
            os.mkdir(os.path.join(config["paths"]["checkpoints"], filename))

    # Data
    train_data = NuScencesMaps(
        config["paths"]["nuscenes"], config["training"]["train_split"]
    )
    trainloader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
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
        num_workers=8,
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

    # Scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, config["training"]["lr_decay"]
    )

    # Reproducability
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # MAIN
    trainer = Trainer(
        model, trainloader, valloader, optimizer, scheduler, criterion, config
    )
    trainer.train()

    # DDP
    destroy_process_group()


if __name__ == "__main__":
    config_name = str(sys.argv[1])
    main(config_name)
