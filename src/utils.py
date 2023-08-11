from collections import defaultdict
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor
import random


def decode_labels(labels, num_classes):
    """
    Decode targets from binary to decimal.

    Args:
        labels (torch.Tensor): Binary targets
        num_classes (int): Number of classes in the decoded target

    Returns:
        labels (torch.Tensor): Decimal targets

    Raises:
        None
    """
    bits = torch.pow(2, torch.arange(num_classes))
    return (labels & bits.view(-1, 1, 1)) > 0


def reduce_labels(labels):
    """
    Reduces NuScenes classes to background/vehicle/person/object

    Args:
        labels (torch.Tensor): Target (14 x H x W)

    Returns:
        labels (torch.Tensor): Target (4 x H x W)

    Raises:
        None
    """
    reduced = torch.zeros((4, labels.shape[1], labels.shape[2]))
    reduced[0, :, :] = labels[0:4, :, :].any(axis=0)  # Drivable area
    reduced[1, :, :] = labels[4:9, :, :].any(axis=0)  # Vehicle
    reduced[2, :, :] = labels[9, :, :]  # Pedestrian
    reduced[3, :, :] = labels[10:, :, :].any(axis=0)  # Other
    return reduced


def upsample_labels(labels, size):
    """
    Upsample labels to desired size.

    Args:
        labels (torch.Tensor): Target (C x H x W)
        size (tuple): (H_new, W_new)

    Returns:
        labels (torch.Tensor): Target (4 x H_nw x W_new)

    Raises:
        None
    """
    labels = labels.unsqueeze(0)
    labels = F.interpolate(labels, size)
    return labels.squeeze(0)


def show_sample(dataset, i):
    """
    Print s sample from a dataset

    Args:
        dataset (torch.utils.data.Dataset): Dataset to plot from
        i (int): Index of sample to plot

    Returns:
        None

    Raises:
        None
    """
    image, _, label, _, mask = dataset[i]

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


def label2image(label: torch.Tensor):
    """
    Convert a target into a colored image.

    Class colors are chosen randomly.

    Args:
        label (troch.Tensor): Label to plot

    Returns:
        Image (torch.Tensor)

    Raises:
        None
    """
    image = torch.zeros((label.shape[1], label.shape[2], 3))
    colors = [
        "AliceBlue",
        "AntiqueWhite",
        "Aqua",
        "Aquamarine",
        "Azure",
        "Beige",
        "Bisque",
        "Black",
        "BlanchedAlmond",
        "Blue",
        "BlueViolet",
        "Brown",
        "BurlyWood",
        "CadetBlue",
        "Chartreuse",
        "Chocolate",
        "Coral",
        "CornflowerBlue",
        "Cornsilk",
        "Crimson",
        "Cyan",
        "DarkBlue",
        "DarkCyan",
        "DarkGoldenRod",
        "DarkGray",
        "DarkGreen",
        "DarkKhaki",
        "DarkMagenta",
        "DarkOliveGreen",
        "DarkOrange",
        "DarkOrchid",
        "DarkRed",
        "DarkSalmon",
        "DarkSeaGreen",
        "DarkSlateBlue",
        "DarkSlateGray",
        "DarkTurquoise",
        "DarkViolet",
        "DeepPink",
        "DeepSkyBlue",
        "DimGray",
        "DodgerBlue",
        "FireBrick",
        "FloralWhite",
        "ForestGreen",
        "Fuchsia",
        "Gainsboro",
        "GhostWhite",
        "Gold",
        "GoldenRod",
        "Gray",
        "Green",
        "GreenYellow",
        "HoneyDew",
        "HotPink",
        "IndianRed",
        "Indigo",
        "Ivory",
        "Khaki",
        "Lavender",
        "LavenderBlush",
        "LawnGreen",
        "LemonChiffon",
        "LightBlue",
        "LightCoral",
        "LightCyan",
        "LightGoldenRodYellow",
        "LightGray",
        "LightGreen",
        "LightPink",
        "LightSalmon",
        "LightSeaGreen",
        "LightSkyBlue",
        "LightSlateGray",
        "LightSteelBlue",
        "LightYellow",
        "Lime",
        "LimeGreen",
        "Linen",
        "Magenta",
        "Maroon",
        "MediumAquaMarine",
        "MediumBlue",
        "MediumOrchid",
        "MediumPurple",
        "MediumSeaGreen",
        "MediumSlateBlue",
        "MediumSpringGreen",
        "MediumTurquoise",
        "MediumVioletRed",
        "MidnightBlue",
        "MintCream",
        "MistyRose",
        "Moccasin",
        "NavajoWhite",
        "Navy",
        "OldLace",
        "Olive",
        "OliveDrab",
        "Orange",
        "OrangeRed",
        "Orchid",
        "PaleGoldenRod",
        "PaleGreen",
        "PaleTurquoise",
        "PaleVioletRed",
        "PapayaWhip",
        "PeachPuff",
        "Peru",
        "Pink",
        "Plum",
        "PowderBlue",
        "Purple",
        "RebeccaPurple",
        "Red",
        "RosyBrown",
        "RoyalBlue",
        "SaddleBrown",
        "Salmon",
        "SandyBrown",
        "SeaGreen",
        "SeaShell",
        "Sienna",
        "Silver",
        "SkyBlue",
        "SlateBlue",
        "SlateGray",
        "Snow",
        "SpringGreen",
        "SteelBlue",
        "Tan",
        "Teal",
        "Thistle",
        "Tomato",
        "Turquoise",
        "Violet",
        "Wheat",
        "White",
        "WhiteSmoke",
        "Yellow",
        "YellowGreen",
    ]

    for i in range(len(label)):
        mask = label[i] == 1
        mask_3d = mask.unsqueeze(dim=0).permute([1, 2, 0]).expand(image.shape)
        color_image = to_tensor(
            Image.new("RGB", (image.shape[0], image.shape[1]), random.choice(colors))
        )
        color_image = torch.permute(color_image, (1, 2, 0))
        image = torch.where(mask_3d, color_image, image)
    return image


def make_grid(grid_size, grid_res):
    """
    Create recitilinear grid.

    From: https://github.com/avishkarsaha/translating-images-into-maps

    Args:
        grid_size (tuple): Size of grid
        grid_res (int): Resolution of grid

    Returns:
        grid (torch.Tensor)

    Raises:
        None
    """
    """
    
    """
    depth, width = grid_size
    xcoords = torch.arange(0.0, width, grid_res)
    zcoords = torch.arange(0.0, depth, grid_res)

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, zz], dim=-1)


class MetricDict(defaultdict):
    """
    Dict to store training metrics.

    From: https://github.com/avishkarsaha/translating-images-into-maps

    """

    def __init__(self):
        super().__init__(float)
        self.count = defaultdict(int)

    def __add__(self, other):
        for key, value in other.items():
            self[key] += value
            self.count[key] += 1
        return self

    @property
    def mean(self):
        return {key: self[key] / self.count[key] for key in self.keys()}
