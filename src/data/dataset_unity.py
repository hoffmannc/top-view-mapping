import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms.functional import to_tensor

from src.utils import make_grid, decode_labels
from src.data.utils_data import get_C902e_parameters, create_vis_mask


class Real(Dataset):
    def __init__(self, path: str, split: str):
        self.path = path
        self.tokens = self.get_tokens(split)

        self.grid_size = (50, 50)
        self.grid_res = 0.5
        self.grid = make_grid(self.grid_size, self.grid_res)

        self.calib, _ = get_C902e_parameters()

        fov = 60
        self.vis_mask = create_vis_mask(fov)

        self.classes = ["drivable_area", "vehicle", "worker", "others"]

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        image = to_tensor(Image.open(f"data/real/samples/{token}.png"))[:3, :, :]
        target = self.get_target(token)
        return image, self.calib, target, self.grid, self.vis_mask

    def get_tokens(self, split):
        path = os.path.join(self.path, "splits", f"{split}.txt")
        with open(path, "r") as f:
            lines = f.read().split("\n")
            return [val for val in lines if val != ""]

    def get_target(self, token):
        target_encoded = to_tensor(Image.open(f"data/real/targets/{token}.png"))
        target_decoded = decode_labels(target_encoded, len(self.classes))
        return torch.flip(target_decoded, [1])
