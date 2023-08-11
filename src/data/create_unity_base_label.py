import numpy as np
import torch
from PIL import Image
from explore_unity_scenario import boundingboxes2map
from utils_data import encode_labels


def main():
    vehicle_keywords = ["Truck", "Trailer", "SM_Veh_Excavator_02"]
    worker_keywords = ["workermodelMesh"]
    object_keywords = ["Prop"]

    temp_vehicle = boundingboxes2map(vehicle_keywords, plot=False)
    temp_worker = boundingboxes2map(worker_keywords, plot=False)
    temp_object = boundingboxes2map(object_keywords, plot=False)

    map = np.stack([temp_vehicle, temp_worker, temp_object])

    temp = np.expand_dims((np.sum(map, axis=0) == 0), axis=0)

    map = np.concatenate([temp, map], axis=0)

    map_vis = boundingboxes2map(
        vehicle_keywords + worker_keywords + object_keywords, True
    )

    labels = encode_labels(torch.tensor(map))

    output_path = "data/unity/base_scenario.png"

    Image.fromarray(labels.numpy().astype(np.int32), mode="I").save(output_path)


if __name__ == "__main__":
    main()
