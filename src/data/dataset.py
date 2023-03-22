import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from PIL import Image, ImageFile
from nuscenes import NuScenes

from src.utils import decode_labels, reduce_labels, downsample


class NuScencesMaps(Dataset):
    def __init__(self, path, split):

        self.path = path

        self.nuscenes = NuScenes("v1.0-trainval", self.path)
        self.tokens = self.get_tokens(split)

        self.image_size = (1280, 720)
        self.grid_size = (50, 50)
        self.grid_res = 0.5
        self.map_size = (100, 100)
        self.grid = self.make_grid()

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
        image = Image.open(self.nuscenes.get_sample_data_path(token))
        calib = self.load_calib(token)
        label, mask = self.load_label(token)
        image, calib = self.image_calib_pad_and_crop(image, calib)
        return image, calib, label, mask

    def load_label(self, token):
        """
        https://github.com/tom-roddick/mono-semantic-maps
        """
        path = os.path.join(self.path, "map-labels-v1.2", token + ".png")
        encoded_labels = to_tensor(Image.open(path)).long()
        num_class = len(self.classes_nuscnenes)
        labels = decode_labels(encoded_labels, num_class + 1)
        labels, mask = labels[:-1], ~labels[-1]
        labels = reduce_labels(labels)
        return labels, mask.double()

    def load_calib(self, token):
        """
        https://github.com/tom-roddick/mono-semantic-maps
        """
        sample_data = self.nuscenes.get("sample_data", token)
        sensor = self.nuscenes.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        return torch.tensor(sensor["camera_intrinsic"])

    def image_calib_pad_and_crop(self, image, calib):
        """
        https://github.com/avishkarsaha/translating-images-into-maps
        """
        og_w, og_h = 1600, 900
        desired_w, desired_h = self.image_size
        scale_w, scale_h = desired_w / og_w, desired_h / og_h
        image = image.resize(
            (int(image.size[0] * scale_w), int(image.size[1] * scale_h))
        )
        w = image.size[0]
        h = image.size[1]
        delta_w = desired_w - w
        delta_h = desired_h - h
        pad_left = int(delta_w / 2)
        pad_right = delta_w - pad_left
        pad_top = int(delta_h / 2)
        pad_bottom = delta_h - pad_top
        left = 0 - pad_left
        right = pad_right + w
        top = 0 - pad_top
        bottom = pad_bottom + h
        image = image.crop((left, top, right, bottom))

        calib[:2, :] *= scale_w
        calib[0, 2] = calib[0, 2] + pad_left
        calib[1, 2] = calib[1, 2] + pad_top

        image = to_tensor(image)

        return image, calib

    def get_tokens(self, split):
        """
        https://github.com/avishkarsaha/translating-images-into-maps
        """
        path = self.split = os.path.join(self.path, "splits", "{}.txt".format(split))
        with open(path, "r") as f:
            lines = f.read().split("\n")
            return [val for val in lines if val != ""]

    def make_grid(self):
        """
        https://github.com/avishkarsaha/translating-images-into-maps
        """
        depth, width = self.grid_size
        xcoords = torch.arange(0.0, width, self.grid_res)
        zcoords = torch.arange(0.0, depth, self.grid_res)

        zz, xx = torch.meshgrid(zcoords, xcoords)
        return torch.stack([xx, zz], dim=-1)
