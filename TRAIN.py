import sys
import os
import yaml

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

from src.data.dataset_nuscenes import NuScencesMaps
import src.model.network as networks


def main(config):
    """
    PyTorch training script for multi-node CUDA training.

    Submit batch job in the LSF10 cluster queue (gpuv100) using /jobs/jobscript.sh.

    Model checkpoints are saved in /checkpoints.
    Training logs are saved in /logs.

    Args:
        config (yaml): Configuration file for the training

    Returns:
        None

    Raises:
        None
    """

    # DDP
    init_process_group()
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Reproducability and CUDA
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)

    # torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    torch.cuda.empty_cache()

    # Configuration
    with open(f"conf/{config_name}.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Files
    if global_rank == 0:
        create_files(config)

    # Data
    train_data = NuScencesMaps(
        config["paths"]["nuscenes"], config["training"]["train_split"]
    )
    trainloader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        # collate_fn=collate,
        drop_last=True,
        pin_memory=False,
        sampler=DistributedSampler(train_data),  # Comment for debugging
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
        # collate_fn=collate,
        pin_memory=False,
        sampler=DistributedSampler(val_data),  # Comment for debugging
    )

    # Model
    model = networks.__dict__[config["modelname"]](**config["model"])
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    # model = DP(model)

    # Loss
    criterion = focal_tversky_loss

    # Optimizer
    optimizer = optim.Adam(model.parameters(), config["training"]["lr"])

    # Scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, config["training"]["lr_decay"]
    )
    # Checkpoint
    epoch = 0
    model, optimizer, scheduler, epoch = load_latest_checkpoint(
        model, optimizer, scheduler, global_rank, epoch, config
    )

    # MAIN
    for epoch in range(epoch, config["training"]["num_epochs"]):
        print(f"[GPU {global_rank}] | E{epoch+1} | Start")

        train(
            model,
            criterion,
            trainloader,
            optimizer,
            global_rank,
            local_rank,
            epoch,
            config,
        )

        if (epoch + 1) % 10 == 0:
            validate(
                model,
                criterion,
                valloader,
                global_rank,
                local_rank,
                epoch,
                config,
            )

        scheduler.step()

        save_checkpoint(model, optimizer, scheduler, global_rank, epoch, config)

    # DDP
    destroy_process_group()


def train(
    model, criterion, trainloader, optimizer, global_rank, local_rank, epoch, config
):
    model.train()
    train_epoch_loss = 0
    for i, (image, calib, target, grid, vis_mask) in enumerate(trainloader):
        image, calib, target, grid, vis_mask = (
            image.float().to(local_rank),
            calib.float().to(local_rank),
            target.float().to(local_rank),
            grid.float().to(local_rank),
            vis_mask.float().to(local_rank),
        )
        outputs = model(image, calib, grid)

        map_sizes = [output.shape[-2:] for output in outputs]
        targets_downsampled = downsample(target, map_sizes)
        loss = compute_loss(outputs, targets_downsampled, criterion)

        train_epoch_loss += loss.item()

        loss.backward()
        if global_rank == 0 and i % 10 == 0:
            plot_grad_flow(model.named_parameters(), config["filename"], epoch, i)
            print(outputs[0])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    document("Training", train_epoch_loss, global_rank, epoch, len(trainloader), config)


def validate(model, criterion, valloader, global_rank, local_rank, epoch, config):
    model.eval()
    val_epoch_loss = 0
    for i, (image, calib, target, grid, vis_mask) in enumerate(valloader):
        image, calib, target, grid, vis_mask = (
            image.float().to(local_rank),
            calib.float().to(local_rank),
            target.float().to(local_rank),
            grid.float().to(local_rank),
            vis_mask.float().to(local_rank),
        )
        with torch.no_grad():
            outputs = model(image, calib, grid)
            map_sizes = [output.shape[-2:] for output in outputs]
            targets_downsampled = downsample(target, map_sizes)
            loss = compute_loss(outputs, targets_downsampled, criterion)

        val_epoch_loss += loss.item()

    document("Validation", val_epoch_loss, global_rank, epoch, len(valloader), config)


def save_checkpoint(model, optimizer, scheduler, global_rank, epoch, config):
    if global_rank == 0:
        checkpoint = {}
        checkpoint["MODEL_STATE"] = model.module.state_dict()
        checkpoint["EPOCHS_RUN"] = epoch
        checkpoint["OPTIMIZER"] = optimizer.state_dict()
        checkpoint["SCHEDULER"] = scheduler.state_dict()
        name = f"{config['filename']}_E{epoch+1}.pt"
        torch.save(
            checkpoint,
            os.path.join(config["paths"]["checkpoints"], config["filename"], name),
        )
        print(f"[GPU {global_rank}] | E{epoch+1} | Saved")


def load_latest_checkpoint(model, optimizer, scheduler, global_rank, epoch, config):
    checkpoints = os.listdir(
        os.path.join(config["paths"]["checkpoints"], config["filename"])
    )
    if checkpoints:
        checkpoints_sorted = sorted(
            checkpoints, key=lambda x: int(x.split("E")[-1].split(".")[0])
        )

        checkpoint = torch.load(
            os.path.join(
                config["paths"]["checkpoints"],
                config["filename"],
                checkpoints_sorted[-1],
            )
        )
        model.module.load_state_dict(checkpoint["MODEL_STATE"])
        optimizer.load_state_dict(checkpoint["OPTIMIZER"])
        scheduler.load_state_dict(checkpoint["SCHEDULER"])
        epoch = checkpoint["EPOCHS_RUN"] + 1

        print(f"[GPU {global_rank}] | E{epoch + 1 - 1} | Loading")

    return model, optimizer, scheduler, epoch


def downsample(target, map_sizes):
    targets_downsampled = []
    t = target
    for size in map_sizes:
        t = F.interpolate(t, size=size, mode="bilinear")
        targets_downsampled.append(t)
    return [
        (target_downsampled > 0).float() for target_downsampled in targets_downsampled
    ]


def dice_loss_mean(pred, label):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1)) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    loss_mean = 1 - iou.mean()

    return loss_mean


def focal_loss(pred, gt, alpha=0.25, gamma=2.0):
    BCE_loss = F.binary_cross_entropy_with_logits(
        pred, gt.float(), reduction="none"
    ).sum(-1)
    gt, _ = torch.max(gt, dim=-1)
    gt = gt.long()
    at = torch.tensor(alpha, device=gt.device).gather(0, gt.data.view(-1))
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt) ** gamma * BCE_loss
    return F_loss


def compute_loss(outputs, targets, criterion):
    ms_loss = torch.stack(
        [criterion(output, target) for output, target in zip(outputs, targets)]
    )
    return torch.sum(ms_loss)


def collate(batch):
    images, calibs, targets, grids, masks = zip(*batch)

    images = torch.stack(images)
    calibs = torch.stack(calibs)
    targets = torch.stack(targets)
    grids = torch.stack(grids)
    masks = torch.stack(masks)

    return images, calibs, targets, grids, masks


def create_files(config):
    filename = config["filename"]
    if not os.path.exists(f"log/{filename}"):
        os.mkdir(f"log/{filename}")
        open(f"log/{filename}/train_loss.txt", "w").close()
        open(f"log/{filename}/val_loss.txt", "w").close()

    if not os.path.exists(os.path.join(config["paths"]["checkpoints"], filename)):
        os.mkdir(os.path.join(config["paths"]["checkpoints"], filename))


def document(mode, loss, global_rank, epoch, n_batches, config):
    print(f"[GPU {global_rank}] | E{epoch+1} | {mode} | {loss / n_batches}")
    with open(
        f"log/{config['filename']}/{'train' if mode == 'Training' else 'val'}_loss.txt",
        "a",
    ) as f:
        f.write(f"GPU{global_rank} E{epoch} " + str((loss / n_batches)) + "\n")


def plot_grad_flow(named_parameters, name, epoch, batch):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    if not os.path.exists(f"figures/{name}"):
        os.mkdir(f"figures/{name}")
    if not os.path.exists(f"figures/{name}/E{epoch}"):
        os.mkdir(f"figures/{name}/E{epoch}")

    ave_grads = []
    max_grads = []
    layers = []
    plt.figure(figsize=(100, 50))
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
            max_grads.append(p.grad.abs().max().cpu().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.savefig(f"figures/{name}/E{epoch}/grad_b{batch}.png")
    plt.close()


def tversky(y_true, y_pred, smooth=1, alpha=0.9):
    y_pred = torch.softmax(y_pred, dim=1)
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_pred, y_true, gamma=5):
    tv = tversky(y_true, y_pred)
    return torch.pow((1 - tv), gamma)


if __name__ == "__main__":
    config_name = str(sys.argv[1])
    main(config_name)
