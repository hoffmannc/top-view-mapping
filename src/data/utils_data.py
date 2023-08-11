import sys
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from scipy.ndimage import rotate  # type: ignore
import cv2


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


def correct_bb(name, w, h):
    if name == "Truck 6":
        return 2.3, 7.7
    elif name == "Truck 1" or name == "Truck 1 (1)":
        return 2.3, 5.5
    elif name == "SM_Veh_Excavator_02":
        return 1.2, 2.0
    else:
        return w, h


def get_C902e_parameters():
    fx = 1394.6027293299926
    fy = 1394.6027293299926
    cx = 995.588675691456
    cy = 599.3212928484164

    M = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    dist = (
        0.11480806073904032,
        -0.21946985653851792,
        0.0012002116999769957,
        0.008564577708855225,
        0.11274677130853494,
    )

    return M, dist


def get_frame(i, file="data/real/drone/drone.mp4"):
    capture = cv2.VideoCapture(file)
    capture.set(cv2.CAP_PROP_POS_FRAMES, i)
    _, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame).save(f"frame_{i}.png")


def create_vis_mask(FOV):
    angle = np.deg2rad(FOV / 2)
    mask = np.zeros((100, 100))

    def R(x):
        return np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])

    o = np.array([50, 0])
    r = R(-angle) @ np.array([0, 200]) + np.array([50, 0])
    l = R(angle) @ np.array([0, 200]) + np.array([50, 0])

    points = np.stack([o, l, r]).astype(int)

    mask = cv2.fillPoly(mask, [points], 1)

    return torch.from_numpy(mask)
