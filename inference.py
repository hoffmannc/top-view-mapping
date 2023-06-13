import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from scipy.ndimage import rotate

from src.utils import label2image, make_grid
from src.model.network import PyrOccTranDetr_S_0904_old_2DPosEncBEV as Model


def main(model_path, image_path, calib_path):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    image = to_tensor(Image.open(image_path))
    calib = torch.load(calib_path)
    grid = make_grid((200, 200), 0.5)
    output = model(image, calib, grid)
    labels = torch.argmax(output, dim=0)
    map = label2image(labels)
    plt.imshow(map)


def combine_outputs(outputs):
    length = len(outputs)
    size = outputs[0].shape
    map = torch.zeros((size[0], 2 * size[1], 2 * size[2]))
    if length == 4:
        angles = [0, 90, 180, 270]
    elif length == 6:
        angles = [0, 60, 120, 180, 240, 300]
    else:
        raise Exception("Invalid number of outputs.")

    for output, angle in zip(outputs, angles):
        map += rotate(output, angle=angle, axes=(1, 2))

    return map


if __name__ == "__main__":
    model_path = sys.argv[0]
    image_path = sys.argv[1]
    calib_path = sys.argv[2]

    main(model_path, image_path, calib_path)
