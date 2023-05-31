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
    # Configuration
    with open("conf/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Data
    train_data = NuScencesMaps(config["paths"]["nuscenes"], "train_split")
    trainloader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        sampler=DistributedSampler(train_data),
    )

    val_data = NuScencesMaps(config["paths"]["nuscenes"], "val_split")
    valloader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        sampler=DistributedSampler(val_data),
    )

    # Model
    model = Model(
        num_classes=config["model"]["num_classes"],
        frontend=config["model"]["frontend"],
        grid_res=config["model"]["grid_res"],
        pretrained=config["model"]["pretrained"],
        img_dims=config["model"]["img_dims"],
        z_range=config["model"]["z_range"],
        h_cropped=config["model"]["h_cropped"],
        dla_norm=config["model"]["dla_norm"],
        additions_BEVT_linear=config["model"]["additions_BEVT_linear"],
        additions_BEVT_conv=config["model"]["additions_BEVT_conv"],
        dla_l1_n_channels=config["model"]["dla_l1_n_channels"],
        n_enc_layers=config["model"]["n_enc_layers"],
        n_dec_layers=config["model"]["n_dec_layers"],
    )

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
        name="Training (Cluster)",
        config=config,
    )

    # MAIN
    init_process_group(backend="nccl")
    trainer = Trainer(model, trainloader, valloader, optimizer, criterion, config)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    main()
