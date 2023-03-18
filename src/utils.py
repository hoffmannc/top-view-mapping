import torch
import numpy as np


def decode_labels(labels, num_classes):
    bits = torch.pow(2, torch.arange(num_classes))
    return (labels & bits.view(-1, 1, 1)) > 0


def reduce_labels(labels):
    reduced = np.zeros((4, 196, 200))
    reduced[0, :, :] = labels[0:4, :, :].any(axis=0)  # Drivable area
    reduced[1, :, :] = labels[4:9, :, :].any(axis=0)  # Vehicle
    reduced[2, :, :] = labels[9, :, :]  # Pedestrian
    reduced[3, :, :] = labels[10:, :, :].any(axis=0)  # Other
    return reduced
