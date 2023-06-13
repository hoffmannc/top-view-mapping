import sys
import json
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


def boundingboxes2map(keywords):
    file = open("data/unity/elements.json")
    elements = json.load(file)

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

    map_size = 200
    resolution = 10

    map = np.zeros((resolution * map_size, resolution * map_size)).astype(bool)

    for element in tqdm(elements, ncols=100):
        name = element["Name"]
        if any(keyword in name for keyword in keywords):
            x = element["X"]
            z = element["Z"]
            angle = element["Rotation_Y"]
            w = element["Size_X"]
            h = element["Size_Z"]
            temp = box2map((x, z, angle), (w, h), map_size, resolution)
            map = map + temp

    plt.imshow(map)
    # plt.axis("equal")
    plt.show()


def points2map(keywords):
    file = open("data/unity/elements.json")
    elements = json.load(file)
    for element in tqdm(elements, ncols=100):
        name = element["Name"]
        if any(keyword in name for keyword in keywords):
            x = element["X"]
            z = element["Z"]
            plt.scatter(x, z, marker=".", c="black")

    plt.axis("equal")
    plt.show()


def main(keywords):
    file = open("data/unity/elements.json")
    elements = json.load(file)
    for element in tqdm(elements, ncols=100):
        name = element["Name"]
        if any(keyword in name for keyword in keywords):
            x = element["X"]
            z = element["Z"]
            plt.scatter(x, z, marker=".", c="black")

    plt.axis("equal")

    fig = plt.gcf()

    coords = []

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print(f"x = {ix}, y = {iy}")

        coords.append((ix, iy))

        return coords

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()


if __name__ == "__main__":
    type = sys.argv[1]
    keywords = sys.argv[2:]
    if type == "b":
        boundingboxes2map(keywords)
    if type == "p":
        points2map(keywords)
    if type == "debug":
        main(keywords)
