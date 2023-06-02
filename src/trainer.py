import os
from typing import Callable
import wandb
from tqdm import tqdm
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        config: dict,
    ):
        self.checkpoints_path = config["paths"]["checkpoints"]

        self.epoch_start = 0

        self.model = model

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self._load_latest_checkpoint()

        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.criterion = criterion

        self.num_epochs = config["training"]["num_epochs"]
        self.batch_size = config["training"]["batch_size"]

        self.filename = config["filename"]

    def _run_epoch(self, epoch: int):
        train_epoch_loss = 0
        for image, calib, target, grid, vis_mask in tqdm(
            self.trainloader,
            desc=f"[GPU {self.gpu_id}] | T | E{epoch+1}",
            total=len(self.trainloader),
            ncols=100,
        ):
            image, calib, target, grid, vis_mask = (
                image.to(torch.float32).to(self.gpu_id),
                calib.to(torch.float32).to(self.gpu_id),
                target.to(torch.float32).to(self.gpu_id),
                grid.to(torch.float32).to(self.gpu_id),
                vis_mask.to(torch.float32).to(self.gpu_id),
            )
            # print(f"[GPU {self.gpu_id}] T{epoch+1} B{i+1}/{len(self.trainloader)}")
            train_batch_loss = self._run_batch(image, calib, target, vis_mask, grid)
            train_epoch_loss += train_batch_loss

        wandb.log(
            {
                "train_epoch_loss": train_epoch_loss
                / (len(self.trainloader) * self.batch_size)
            }
        )

    def _run_batch(
        self,
        image: torch.Tensor,
        calib: torch.Tensor,
        target: torch.Tensor,
        vis_mask: torch.Tensor,
        grid: torch.Tensor,
    ):
        self.optimizer.zero_grad()
        outputs = self.model(image, calib, grid)

        map_sizes = [output.shape[-2:] for output in outputs]
        targets_downsampled = self._downsample(target, map_sizes)
        loss = self._compute_loss(outputs, targets_downsampled)

        loss.backward()
        self.optimizer.step()
        return loss

    def _downsample(self, target, map_sizes):
        targets_downsampled = []
        targets_downsampled.append(target)
        for size in map_sizes[1:]:
            t = F.interpolate(target, size=size, mode="bilinear")
            targets_downsampled.append(t)
        return targets_downsampled

    def _compute_loss(self, outputs, labels):
        ms_loss = torch.stack(
            [self.criterion(output, label) for output, label in zip(outputs, labels)]
        )
        total_loss = torch.sum(ms_loss)
        return total_loss

    def _eval_epoch(self, epoch: int):
        val_epoch_loss = 0
        for image, calib, target, grid, vis_mask in tqdm(
            self.valloader,
            desc=f"[GPU {self.gpu_id}] | V | E{epoch+1}",
            total=len(self.valloader),
            ncols=100,
        ):
            image, calib, target, grid, vis_mask = (
                image.to(torch.float32).to(self.gpu_id),
                calib.to(torch.float32).to(self.gpu_id),
                target.to(torch.float32).to(self.gpu_id),
                grid.to(torch.float32).to(self.gpu_id),
                vis_mask.to(torch.float32).to(self.gpu_id),
            )
            val_batch_loss = self._eval_batch(image, calib, target, vis_mask, grid)
            val_epoch_loss += val_batch_loss

        wandb.log(
            {"val_epoch_loss": val_epoch_loss / (len(self.valloader) * self.batch_size)}
        )

    def _eval_batch(
        self,
        image: torch.Tensor,
        calib: torch.Tensor,
        target: torch.Tensor,
        vis_mask: torch.Tensor,
        grid: torch.Tensor,
    ):
        with torch.no_grad():
            outputs = self.model(image, calib, grid)
            map_sizes = [output.shape[-2:] for output in outputs]
            targets_downsampled = self._downsample(target, map_sizes)
            loss = self._compute_loss(outputs, targets_downsampled)
        return loss

    def _save_checkpoint(self, epoch: int):
        if self.gpu_id == 0:
            checkpoint = {}
            checkpoint["MODEL_STATE"] = self.model.module.state_dict()
            checkpoint["EPOCHS_RUN"] = epoch
            name = f"/{self.filename}_E{epoch+1}.pt"
            torch.save(checkpoint, self.checkpoints_path + name)
            print(f"[GPU {self.gpu_id}] | S | E{epoch+1}")

    def _load_checkpoint(self, name):
        checkpoint = torch.load(os.path.join(self.checkpoints_path, name))
        self.model.module.load_state_dict(checkpoint["MODEL_STATE"])
        epochs_run = checkpoint["EPOCHS_RUN"]
        self.epoch_start = epochs_run + 1
        print(f"[GPU {self.gpu_id}] | L | E{epochs_run + 1}")

    def _load_latest_checkpoint(self):
        checkpoints = os.listdir(self.checkpoints_path)
        if checkpoints:
            checkpoints_sorted = sorted(checkpoints)
            self._load_checkpoint(checkpoints_sorted[-1])

    def train(self):
        for epoch in range(self.epoch_start, self.num_epochs):
            self.model.train()
            self._run_epoch(epoch)
            self.model.eval()
            self._eval_epoch(epoch)
            self._save_checkpoint(epoch)
