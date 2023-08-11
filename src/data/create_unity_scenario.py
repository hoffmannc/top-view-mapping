import sys
import json
import matplotlib.pyplot as plt
import numpy as np


def main(id):
    """
    Creates scenario for the unity dataset creation

    Store the data in the root.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    file = open("data/unity/elements.json")
    elements = json.load(file)

    file = open("data/unity/scenario_types.json")
    objects = json.load(file)

    alias_types = ["Trajectory", "Excavator", "Avatar"]

    scenario = []

    # Orientation points
    x, z = [], []
    keywords = ["Fence", "Barrier", "Barrel"]
    for element in elements:
        name = element["Name"]
        if any(keyword in name for keyword in keywords):
            x.append(element["X"])
            z.append(element["Z"])

    fig = plt.figure(figsize=(8, 8))
    plt.title("Scenario")
    plt.scatter(x, z, marker=".", c="black")  # type: ignore
    plt.axis("equal")
    plt.show(block=False)
    plt.ion()

    coords = []

    colors = iter(("red", "blue", "green"))

    # Objects
    for alias in alias_types:
        coords = []
        plt.title(alias)
        color = next(colors)

        def onclick(event):
            ix, iy = event.xdata, event.ydata

            coords.append((round(ix, 2), round(iy, 2)))

            plt.scatter(ix, iy, marker=".", c=color)

            return coords

        plt.draw()
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        keyboardClick = False
        while keyboardClick != True:
            keyboardClick = plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(cid)

        if alias == "Trajectory":
            plt.plot(
                [coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]], color=color
            )

            item = objects[0].copy()
            item["X"], item["Z"] = coords[0][0], coords[0][1]
            item["X - Trajectory"], item["Z - Trajectory"] = coords[1][0], coords[1][1]
            angle = np.arctan2(
                (item["Z - Trajectory"] - item["Z"]),
                (item["X - Trajectory"] - item["X"]),
            )
            item["Rotation_Y"] = np.degrees(angle)
            d = np.sqrt(
                (item["X - Trajectory"] - item["X"]) ** 2
                + (item["Z - Trajectory"] - item["Z"]) ** 2
            )
            v = 20 / 3.6
            t = d / v
            item["Duration"] = t
            scenario.append(item)

        elif alias == "Excavator":
            for i, (x, z) in enumerate(coords):
                item = objects[1].copy()
                item["Name"] = item["Name"] + f"_{i+1:02d}"
                item["X"] = x
                item["Z"] = z
                item["Rotation_Y"] = np.random.randint(0, 359)
                scenario.append(item)
        elif alias == "Avatar":
            for i, (x, z) in enumerate(coords):
                item = objects[2].copy()
                item["Name"] = item["Name"] + f"_{i+1:02d}"
                item["X"] = x
                item["Z"] = z
                item["Rotation_Y"] = np.random.randint(0, 359)
                scenario.append(item)

    save_name = f"data/unity/scenarios/{id}.json"

    with open(save_name, "w") as fp:
        json.dump(scenario, fp)


if __name__ == "__main__":
    id = sys.argv[1]
    main(id)
