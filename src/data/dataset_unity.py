import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms.functional import to_tensor
from src.data.utils_data import create_vis_mask
from src.utils import make_grid, decode_labels


class Unity(Dataset):
    def __init__(self, path: str, split: str):
        self.path = path
        self.tokens = self.get_tokens(split)

        self.grid_size = (50, 50)
        self.grid_res = 0.5
        self.grid = make_grid(self.grid_size, self.grid_res)

        focal_length = 50
        sensor_size = (36, 24)
        image_size = (1600, 900)
        optical_center = (image_size[0] / 2, image_size[1] / 2)
        pixel_size = (sensor_size[0] / image_size[0], sensor_size[1] / image_size[1])

        self.calib = torch.tensor(
            [
                [focal_length / pixel_size[0], 0, optical_center[0]],
                [0, focal_length / pixel_size[1], optical_center[1]],
                [0, 0, 1],
            ]
        )

        fov = 90
        self.vis_mask = create_vis_mask(fov)

        self.classes = ["drivable_area", "vehicle", "worker", "others"]

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def len(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        image = to_tensor(Image.open(f"data/unity/samples/{token}.png"))[:3, :, :]
        target = self.get_target(token)
        return image, self.calib, target, self.grid, self.vis_mask

    def get_target(self, token):
        target_encoded = to_tensor(Image.open(f"data/unity/targets/{token}.png"))
        target_decoded = decode_labels(target_encoded, len(self.classes))
        return torch.flip(target_decoded, [1])

    def get_tokens(self, split):
        """
        https://github.com/avishkarsaha/translating-images-into-maps
        """
        path = os.path.join(self.path, "splits", "{}.txt".format(split))
        with open(path, "r") as f:
            lines = f.read().split("\n")
            return [val for val in lines if val != ""]
