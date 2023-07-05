import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from scipy.ndimage import rotate  # type: ignore


def decode_labels(labels, num_classes):
    bits = torch.pow(2, torch.arange(num_classes))
    return (labels & bits.view(-1, 1, 1)) > 0


def encode_labels(labels):
    bits = torch.pow(2, torch.arange(len(labels), dtype=torch.int32))
    return (labels.type(torch.int32) * bits.reshape(-1, 1, 1)).sum(0)


def box2map(pos, shape, map_size, resolution):
    temp = np.zeros((map_size * resolution, map_size * resolution)).astype(bool)
    const = int(map_size * resolution / 2)
    x, y, angle = pos
    w, h = shape
    l = const - int(w * resolution / 2)
    r = const + int(w * resolution / 2)
    t = const - int(h * resolution / 2)
    b = const + int(h * resolution / 2)
    temp[t:b, l:r] = True
    temp = rotate(temp, angle=90 - angle, reshape=False, order=0)
    map = np.roll(
        temp,
        shift=(int(x * resolution), int(y * resolution)),
        axis=(0, 1),
    ).astype(bool)
    return map.astype(bool)


def label2image(label: torch.Tensor):
    image = torch.zeros((label.shape[1], label.shape[2], 3))
    colors = ["white", "blue", "green", "red"]
    for i in range(4):
        mask = label[i] == 1
        mask_3d = mask.unsqueeze(dim=0).permute([1, 2, 0]).expand(image.shape)
        color_image = to_tensor(
            Image.new("RGB", (image.shape[0], image.shape[1]), colors[i])
        )
        color_image = torch.permute(color_image, (1, 2, 0))
        image = torch.where(mask_3d, color_image, image)
    return image
