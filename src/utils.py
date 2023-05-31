import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.distributed import init_process_group, destroy_process_group


def decode_labels(labels, num_classes):
    bits = torch.pow(2, torch.arange(num_classes))
    return (labels & bits.view(-1, 1, 1)) > 0


def reduce_labels(labels):
    reduced = torch.zeros((4, labels.shape[1], labels.shape[2]))
    reduced[0, :, :] = labels[0:4, :, :].any(axis=0)  # Drivable area
    reduced[1, :, :] = labels[4:9, :, :].any(axis=0)  # Vehicle
    reduced[2, :, :] = labels[9, :, :]  # Pedestrian
    reduced[3, :, :] = labels[10:, :, :].any(axis=0)  # Other
    return reduced


def upsample_labels(labels, size):
    labels = labels.unsqueeze(0)
    labels = F.interpolate(labels, size)
    return labels.squeeze(0)


def show_sample(dataset, i):
    image, _, label, mask = dataset[i]

    _, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].imshow(image.permute(1, 2, 0))
    ax[0, 1].imshow(mask.squeeze(0), origin="lower")
    ax[0, 2].imshow(label[0, :, :], origin="lower")
    ax[1, 0].imshow(label[1, :, :], origin="lower")
    ax[1, 1].imshow(label[2, :, :], origin="lower")
    ax[1, 2].imshow(label[3, :, :], origin="lower")

    ax[0, 0].tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    ax[0, 1].tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    ax[0, 2].tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    ax[1, 0].tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    ax[1, 1].tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    ax[1, 2].tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    plt.show()


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


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print("--- Process group set up ---")


def ddp_destroy():
    destroy_process_group()
    print("--- Process group destroyed ---")


def label2image(label: torch.Tensor) -> torch.Tensor:
    image = torch.zeros((label.shape[1], label.shape[2], 3))
    colors = ["blue", "red", "green", "yellow"]
    for i in range(4):
        mask = label[i] == 1
        mask_3d = mask.unsqueeze(2).expand(image.shape)
        color_image = to_tensor(
            Image.new("RGB", (image.shape[0], image.shape[1]), colors[i])
        )
        color_image = torch.permute(color_image, (1, 2, 0))
        image = torch.where(mask_3d, color_image, image)
    return image
