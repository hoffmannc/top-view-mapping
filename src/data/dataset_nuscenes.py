import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageFile
from nuscenes import NuScenes

from src.utils import decode_labels, make_grid


class NuScencesMaps(Dataset):
    def __init__(self, path: str, split: str):
        self.path = path
        self.nuscenes = NuScenes("v1.0-trainval", self.path)
        self.tokens = self.get_tokens(split)

        self.image_size = (1600, 900)
        self.grid_size = (50, 50)
        self.grid_res = 0.5
        self.map_size = (200, 200)
        self.grid = make_grid(self.grid_size, self.grid_res)

        self.classes_nuscnenes = [
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark",
            "car",
            "truck",
            "bus",
            "trailer",
            "construction_vehicle",
            "pedestrian",
            "motorcycle",
            "bicycle",
            "traffic_cone",
            "barrier",
        ]

        self.classes = ["drivable_area", "vehicle", "pedestrian", "others"]

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        image = to_tensor(Image.open(self.nuscenes.get_sample_data_path(token)))
        calib = self.load_calib(token)
        label, mask = self.load_label(token)
        return image, calib, label, self.grid, mask

    def load_label(self, token):
        """
        https://github.com/tom-roddick/mono-semantic-maps
        """
        path = os.path.join(self.path, "targets", token + ".png")
        labels = to_tensor(Image.open(path)).long()
        labels = decode_labels(labels, 4 + 1)
        labels, mask = labels[:-1], ~labels[-1]
        return labels.double(), mask.double()

    def load_calib(self, token):
        """
        https://github.com/tom-roddick/mono-semantic-maps
        """
        sample_data = self.nuscenes.get("sample_data", token)
        sensor = self.nuscenes.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        return torch.tensor(sensor["camera_intrinsic"])

    def get_tokens(self, split):
        """
        https://github.com/avishkarsaha/translating-images-into-maps
        """
        path = self.split = os.path.join(self.path, "splits", "{}.txt".format(split))
        with open(path, "r") as f:
            lines = f.read().split("\n")
            return [val for val in lines if val != ""]
