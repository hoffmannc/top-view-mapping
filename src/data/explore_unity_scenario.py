import sys
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_data import box2map


def boundingboxes2map(keywords, plot=False):
    file = open("data/unity/elements.json")
    elements = json.load(file)

    map_size = 300
    resolution = 20

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

    if plot:
        plt.imshow(map)
        plt.show()
    return map


def points2map(keywords, plot=False):
    file = open("data/unity/elements.json")
    elements = json.load(file)
    for element in tqdm(elements, ncols=100):
        name = element["Name"]
        if any(keyword in name for keyword in keywords):
            x = element["X"]
            z = element["Z"]
            plt.scatter(x, z, marker=".", c="black")  # type: ignore

    if plot:
        plt.axis("equal")
        plt.show()


def main(keywords, plot):
    file = open("data/unity/elements.json")
    elements = json.load(file)
    for element in tqdm(elements, ncols=100):
        name = element["Name"]
        if any(keyword in name for keyword in keywords):
            x = element["X"]
            z = element["Z"]
            plt.scatter(x, z, marker=".", c="black")  # type: ignore

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
    keywords = sys.argv[3:]
    plot = sys.argv[2] == "True"
    if type == "b":
        boundingboxes2map(keywords, plot)
    if type == "p":
        points2map(keywords, plot)
    if type == "debug":
        main(keywords, None)
