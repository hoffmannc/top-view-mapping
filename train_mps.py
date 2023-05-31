import yaml
import wandb

import torch
import torch.nn as nn
from torch import optim, mps
from torch.utils.data import DataLoader

from src.data.dataset import NuScencesMaps
from src.model.network import PyrOccTranDetr_S_0904_old_2DPosEncBEV as Model
from src.utils import dice_loss_mean


def train(model, dataloader, optimizer, criterion, metrics, device):
    model.train()
    loss_epoch = 0

    for i, (image, calib, target, grid, vis_mask) in enumerate(dataloader):
        image, calib, target, vis_mask, grid = (
            image.to(torch.float32).to(device),
            calib.to(torch.float32).to(device),
            target.to(torch.float32).to(device),
            vis_mask.to(torch.float32).to(device),
            grid.to(torch.float32).to(device),
        )
        prediction = model(image, calib, grid)

        target = (target > 0).float()

        loss = criterion(prediction[0], target)

        loss_epoch += loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        print("Batch " + str(i + 1))

    metrics["train_loss"].append(loss_epoch)


def validate(model, dataloader, criterion, metrics, device):
    model.eval()

    total_loss = 0

    for i, (image, calib, target, grid, vis_mask) in enumerate(dataloader):
        image, calib, target, vis_mask, grid = (
            image.to(device),
            calib.to(device),
            target.to(device),
            vis_mask.to(device),
            grid.to(device),
        )

        with torch.no_grad():
            prediction = model(image, calib, grid)

            loss = criterion(prediction, target)

            total_loss += loss

    metrics["val_loss"].append(total_loss)


def main():
    # Configuration
    with open("conf/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Reproducability
    seed = 42
    torch.manual_seed(seed)
    mps.manual_seed(seed)

    # WandB
    wandb.init(project="top-view-mapping", entity="hoffmann", config=config)

    # Data
    train_data = NuScencesMaps(config["paths"]["nuscenes"], "train_split")
    train_loader = DataLoader(
        train_data,
        batch_size=wandb.config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    val_data = NuScencesMaps(config["paths"]["nuscenes"], "val_split")
    val_loader = DataLoader(
        val_data,
        batch_size=wandb.config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    # Model
    model = Model(
        num_classes=wandb.config["model"]["num_classes"],
        frontend=wandb.config["model"]["frontend"],
        grid_res=wandb.config["model"]["grid_res"],
        pretrained=wandb.config["model"]["pretrained"],
        img_dims=wandb.config["model"]["img_dims"],
        z_range=wandb.config["model"]["z_range"],
        h_cropped=wandb.config["model"]["h_cropped"],
        dla_norm=wandb.config["model"]["dla_norm"],
        additions_BEVT_linear=wandb.config["model"]["additions_BEVT_linear"],
        additions_BEVT_conv=wandb.config["model"]["additions_BEVT_conv"],
        dla_l1_n_channels=wandb.config["model"]["dla_l1_n_channels"],
        n_enc_layers=wandb.config["model"]["n_enc_layers"],
        n_dec_layers=wandb.config["model"]["n_dec_layers"],
    )

    device = torch.device("mps")
    model = model.to(device)

    # Loss function
    criterion = dice_loss_mean

    # Optimizer
    optimizer = optim.Adam(model.parameters(), wandb.config["training"]["lr"])

    # Metrics
    metrics = {"train_loss": [], "val_loss": []}

    mps.empty_cache()

    for e in range(wandb.config["training"]["num_epochs"]):
        print("Epoch " + str(e + 1))

        train(model, train_loader, optimizer, criterion, metrics, device)
        validate(model, val_loader, criterion, metrics, device)

        torch.save(
            model.state_dict(), config["paths"]["checkpoints"] + "_" + str(e) + ".pt"
        )

        wandb.log(metrics)


if __name__ == "__main__":
    main()
