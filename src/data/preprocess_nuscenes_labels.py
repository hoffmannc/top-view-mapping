import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F


def decode_labels(labels, num_classes):
    bits = torch.pow(2, torch.arange(num_classes))
    return (labels & bits.view(-1, 1, 1)) > 0


def encode_labels(labels):
    bits = torch.pow(2, torch.arange(len(labels), dtype=torch.int32))
    return (labels.type(torch.int32) * bits.reshape(-1, 1, 1)).sum(0)


def reduce_labels(labels):
    reduced = torch.zeros((5, 196, 200))
    reduced[0, :, :] = labels[0:4, :, :].any(axis=0)  # Drivable area
    reduced[1, :, :] = labels[4:9, :, :].any(axis=0)  # Vehicle
    reduced[2, :, :] = labels[9, :, :]  # Pedestrian
    reduced[3, :, :] = labels[10:14, :, :].any(axis=0)  # Other
    reduced[4, :, :] = labels[14, :, :]  # Mask
    return reduced


target_path = "/Users/colinhoffmann/Desktop/top-view-mapping/nuscenes/targets/"
input_path = "/Users/colinhoffmann/Desktop/top-view-mapping/nuscenes/map-labels-v1.2/"

for i, filename in enumerate(os.listdir(input_path)):
    path = os.path.join(input_path, filename)
    encoded_labels = to_tensor(Image.open(path)).long()
    num_class = 14 + 1
    labels = decode_labels(encoded_labels, num_class)
    labels = reduce_labels(labels)
    labels = labels[:, :, 3:199]
    size = (100, 100)
    labels = labels.unsqueeze(0)
    labels = F.interpolate(labels, size)
    labels = labels.squeeze(0)
    labels = encode_labels(labels)
    output_path = target_path + filename
    Image.fromarray(labels.numpy().astype(np.int32), mode="I").save(output_path)

    if i % 1000 == 0:
        print(i)
