import os
import json
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import interpolate
from torchvision.transforms.functional import rotate
from scipy.ndimage import shift

from explore_unity_scenario import boundingboxes2map
from utils_data import decode_labels, encode_labels, box2map, label2image


def main():
    """
    Creates targets for unity dataset.

    Store the data in the root.
    Run create_unity_base_label.py before.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    n_scenarios = len(os.listdir("data/unity/scenarios"))
    base_scenario_path = "data/unity/base_scenario.png"
    base_scenario = to_tensor(Image.open(base_scenario_path)).long()
    base_scenario = decode_labels(base_scenario, 4)

    map_size = 300
    resolution = 20

    for i in range(n_scenarios):
        map = base_scenario.clone()
        name = f"scenario{i+1}"
        file = open(f"data/unity/scenarios/{name}.json")
        elements = json.load(file)

        for element in elements[1:]:
            x = element["X"]
            z = element["Z"]
            angle = element["Rotation_Y"]
            w = element["Size_X"]
            h = element["Size_Z"]
            temp = box2map((x, z, angle), (w, h), map_size, resolution)
            if "Excavator" in element["Name"]:
                map[1, :, :] += temp
            elif "Worker" in element["Name"]:
                map[2, :, :] += temp

        yaw = elements[0]["Rotation_Y"]

        file = open(f"data/unity/scenarios_generated/{name}/scenario.json")
        steps = json.load(file)

        for step in tqdm(steps):
            frame_id = step["FrameId"]
            ego = step["Elements"][-1]

            x = ego["X"]
            z = ego["Z"]
            angle = yaw

            temp = map.clone()
            shifted = shift(
                temp,
                shift=(0, -int(x * resolution), -int(z * resolution)),
                order=0,
            )

            view_angles = [0, 90, 180, 270]
            views = ["Front", "Right", "Back", "Left"]

            for view_angle, view in zip(view_angles, views):
                rotated = rotate(
                    torch.Tensor(shifted).clone(),
                    angle=angle + 90 + view_angle,
                )

                const = int(rotated.shape[1] / 2)

                cut = rotated[
                    :,
                    const - 50 * resolution : const,
                    const - 25 * resolution : const + 25 * resolution,
                ]

                label = interpolate(
                    torch.Tensor(cut).unsqueeze(dim=0), size=(100, 100)
                ).squeeze(dim=0)

                final_label = encode_labels(torch.Tensor(label))
                output_path = f"data/unity/targets/{frame_id}_{view}.png"
                Image.fromarray(final_label.numpy().astype(np.int32), mode="I").save(
                    output_path
                )


main()
