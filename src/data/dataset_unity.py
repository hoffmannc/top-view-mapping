import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms.functional import to_tensor

from src.utils import make_grid, decode_labels


class Unity(Dataset):
    def __init__(self, path: str, split: str):
        self.path = path
        self.tokens = self.get_tokens(split)

        self.grid_size = (50, 50)
        self.grid_res = 0.5
        self.grid = make_grid(self.grid_size, self.grid_res)

        self.calib = None

        self.vis_mask = None

        self.classes = ["drivable_area", "vehicle", "worker", "others"]

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def len(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        image = to_tensor(Image.open(f"data/unity/samples/{token}.png"))
        target = self.get_target(token)
        return image, self.calib, target, self.grid, self.vis_mask

    def get_target(self, token):
        target_encoded = to_tensor(Image.open(f"data/unity/targets/{token}.png"))
        return decode_labels(target_encoded, len(self.classes))

    def get_tokens(self, split):
        """
        https://github.com/avishkarsaha/translating-images-into-maps
        """
        path = self.split = os.path.join(self.path, "splits", "{}.txt".format(split))
        with open(path, "r") as f:
            lines = f.read().split("\n")
            return [val for val in lines if val != ""]
