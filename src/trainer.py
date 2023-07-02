import os
from typing import Callable
from tqdm import tqdm

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
        scheduler: torch.optim.lr_scheduler.ExponentialLR,
        criterion: Callable,
        config: dict,
    ):
        self.checkpoints_path = config["paths"]["checkpoints"]
        self.filename = config["filename"]

        self.epoch_start = 0

        self.model = model

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.num_epochs = config["training"]["num_epochs"]
        self.batch_size = config["training"]["batch_size"]

        self._load_latest_checkpoint()

    def _run_epoch(self, epoch: int):
        train_epoch_loss = 0
        for i, (image, calib, target, grid, vis_mask) in enumerate(self.trainloader):
            image, calib, target, grid, vis_mask = (
                image.to(torch.float32).to(self.local_rank),
                calib.to(torch.float32).to(self.local_rank),
                target.to(torch.float32).to(self.local_rank),
                grid.to(torch.float32).to(self.local_rank),
                vis_mask.to(torch.float32).to(self.local_rank),
            )
            train_batch_loss = self._run_batch(image, calib, target, vis_mask, grid)
            train_epoch_loss += train_batch_loss

        print(
            f"[GPU {self.global_rank}] | E{epoch+1} | Train | {train_epoch_loss/(len(self.trainloader) * self.batch_size)}"
        )
        with open(f"log/{self.filename}/train_loss.txt", "a") as f:
            f.write(
                f"GPU{self.global_rank} E{epoch} "
                + str(
                    (
                        train_epoch_loss / (len(self.trainloader) * self.batch_size)
                    ).item()
                )
                + "\n"
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
        for i, (image, calib, target, grid, vis_mask) in enumerate(self.valloader):
            image, calib, target, grid, vis_mask = (
                image.to(torch.float32).to(self.local_rank),
                calib.to(torch.float32).to(self.local_rank),
                target.to(torch.float32).to(self.local_rank),
                grid.to(torch.float32).to(self.local_rank),
                vis_mask.to(torch.float32).to(self.local_rank),
            )
            val_batch_loss = self._eval_batch(image, calib, target, vis_mask, grid)
            val_epoch_loss += val_batch_loss

        print(
            f"[GPU {self.global_rank}] | E{epoch+1} | Validate | {val_epoch_loss / (len(self.valloader) * self.batch_size)}"
        )
        with open(f"log/{self.filename}/val_loss.txt", "a") as f:
            f.write(
                f"GPU{self.global_rank} E{epoch} "
                + str((val_epoch_loss / (len(self.valloader) * self.batch_size)).item())
                + "\n"
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
        if self.global_rank == 0:
            checkpoint = {}
            checkpoint["MODEL_STATE"] = self.model.module.state_dict()
            checkpoint["EPOCHS_RUN"] = epoch
            checkpoint["OPTIMIZER"] = self.optimizer.state_dict()
            checkpoint["SCHEDULER"] = self.scheduler.state_dict()
            name = f"{self.filename}_E{epoch+1}.pt"
            torch.save(
                checkpoint, os.path.join(self.checkpoints_path, self.filename, name)
            )
            print(f"[GPU {self.global_rank}] | E{epoch+1} | Save")

    def _load_checkpoint(self, name):
        checkpoint = torch.load(
            os.path.join(self.checkpoints_path, self.filename, name)
        )
        self.model.module.load_state_dict(checkpoint["MODEL_STATE"])
        epochs_run = checkpoint["EPOCHS_RUN"]
        self.optimizer.load_state_dict(checkpoint["OPTIMIZER"])
        self.scheduler.load_state_dict(checkpoint["SCHEDULER"])
        self.epoch_start = epochs_run + 1
        print(f"[GPU {self.global_rank}] | E{epochs_run + 1} | Load")

    def _load_latest_checkpoint(self):
        checkpoints = os.listdir(os.path.join(self.checkpoints_path, self.filename))
        if checkpoints:
            checkpoints_sorted = sorted(
                checkpoints, key=lambda x: int(x.split("E")[-1].split(".")[0])
            )
            self._load_checkpoint(checkpoints_sorted[-1])

    def train(self):
        for epoch in range(self.epoch_start, self.num_epochs):
            print(f"[GPU {self.global_rank}] | E{epoch+1} | Start")
            self.model.train()
            self._run_epoch(epoch)
            self.model.eval()
            self._eval_epoch(epoch)
            self.scheduler.step()
            self._save_checkpoint(epoch)
