import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def decode_labels(labels, num_classes):
    bits = torch.pow(2, torch.arange(num_classes))
    return (labels & bits.view(-1, 1, 1)) > 0


def reduce_labels(labels):
    reduced = torch.zeros((4, 196, 200))
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
