import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import matplotlib.pyplot as plt

from src.data.dataset import NuScencesMaps
from src.utils import decode_labels

train_data = NuScencesMaps("nuscenes", "train_split")


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


label = train_data[0][2]
image = label2image(label)

test = 0
