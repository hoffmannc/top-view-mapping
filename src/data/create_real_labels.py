import os
import json
from itertools import combinations

import numpy as np
import torch
import cv2
from PIL import Image
from torch.nn.functional import interpolate
from scipy.ndimage import shift, rotate


def encode_labels(labels):
    bits = torch.pow(2, torch.arange(len(labels), dtype=torch.int32))
    return (labels.type(torch.int32) * bits.reshape(-1, 1, 1)).sum(0)


def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_origin(points):
    combis = list(combinations(list(range(len(points))), 2))
    distances = []
    for combi in combis:
        i1, i2 = combi
        p1, p2 = points[i1], points[i2]
        distances.append(get_distance(p1, p2))
    v1, v2 = np.argsort(distances)[:2]

    i_corner = (set(combis[v1]) & set(combis[v2])).pop()
    corner = points[i_corner]

    d = []
    for p in points:
        p1 = points[i_corner]
        d.append(get_distance(p1, p))

    i_edge1, i_edge2 = np.argsort(d)[-2:]
    edge1 = points[i_edge1]
    edge2 = points[i_edge2]
    corner = points[i_corner]
    edge1 = points[i_edge1]
    edge2 = points[i_edge2]

    middle = edge1 + (edge2 - edge1) / 2

    origin = corner + (middle - corner) * 0.75

    vec = middle - corner

    angle = np.degrees(np.arctan2(-vec[1], vec[0]))

    angle = (angle + 360) % 360

    return origin, angle


def transform_points(points_raw, width, height):
    points_transformed = []
    for x, y in points_raw:
        points_transformed.append((int(x / 100 * width), int(y / 100 * height)))
    return np.array(points_transformed)


def sample2label(sample):
    token = sample["file_upload"].split("-")[1].split(".")[0]

    file = open("data/real/metadata.json")
    meta = json.load(file)

    height, width = 2160, 3840

    view_dict = {
        "LEFT": 54,
        "MIDLEFT": 18,
        "MIDRIGHT": -18,
        "RIGHT": -54,
    }

    views = [s["cam_view"] for s in meta if s["drone_token"] == token]
    tokens_samples = [s["token"] for s in meta if s["drone_token"] == token]

    annotations = sample["annotations"][0]["result"]
    label = np.zeros((4, height, width))
    for annotation in annotations:
        type = annotation["value"]["polygonlabels"][0]
        if type == "Camera mount":
            points_final = transform_points(
                annotation["value"]["points"], width, height
            )
            origin, angle = get_origin(points_final)
        else:
            temp = np.zeros((height, width))
            points_final = transform_points(
                annotation["value"]["points"], width, height
            )
            temp = cv2.fillPoly(temp, pts=[points_final], color=1)
            if type == "Vehicle":
                label_dim = 1
            elif type == "Person":
                label_dim = 2
            elif type == "Object":
                label_dim = 3
            else:
                raise Exception
            label[label_dim, :, :] += temp

    label = torch.Tensor(label)

    pad = ((0, 0), (height, 0), (width, 0))
    padded = np.pad(label, pad, constant_values=0.0)

    shifted = shift(padded, shift=(0, -origin[1], -origin[0]), order=0, cval=0.0)  # type: ignore

    const_pixel = 175 / 4.07

    for token, view in zip(tokens_samples, views):
        rotated = rotate(
            shifted.copy(), angle=-(angle + view_dict[view]), axes=(1, 2), order=0, reshape=False  # type: ignore
        )

        temp = (rotated.sum(axis=0) == 0).astype(float)
        rotated[0, :, :] = temp

        cut = rotated[
            :,
            int(height - 25 * const_pixel) : int(height + 25 * const_pixel + 1),
            int(width) : int(width + 50 * const_pixel + 1),
        ]

        label = interpolate(
            torch.Tensor(cut).unsqueeze(dim=0), size=(100, 100)
        ).squeeze(dim=0)

        label = torch.rot90(label, 1, [1, 2])

        final_label = encode_labels(torch.Tensor(label))
        output_path = f"data/real/targets/{token}.png"
        Image.fromarray(final_label.numpy().astype(np.int32), mode="I").save(
            output_path
        )
        with open("data/real/splits/all.txt", "a") as f:
            f.write(f"{token}\n")


def main():
    projects = os.listdir("data/real/drone/annotation_projects")
    projects = [project for project in projects if project.split(".")[-1] == "json"]

    projects_done = (
        open("data/real/drone/annotation_projects/done.txt", "r").read().split("\n")
    )

    for project in projects:
        if project in projects_done:
            continue
        else:
            file = open(os.path.join("data/real/drone/annotation_projects", project))
            samples = json.load(file)
            for sample in samples:
                sample2label(sample)

        with open("data/real/drone/annotation_projects/done.txt", "a") as f:
            f.write(f"{project}\n")


if __name__ == "__main__":
    main()
