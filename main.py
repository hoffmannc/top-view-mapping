from src.data.dataset import NuScencesMaps
import matplotlib.pyplot as plt


def show_sample(i):
    image, _, label, mask = dataset[i]

    _, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].imshow(image.permute(1, 2, 0))
    ax[0, 1].imshow(mask)
    ax[0, 2].imshow(label[0, :, :])
    ax[1, 0].imshow(label[1, :, :])
    ax[1, 1].imshow(label[2, :, :])
    ax[1, 2].imshow(label[3, :, :])
    plt.show()


path = "/Users/colinhoffmann/Desktop/nuscenes"
split = "train_split"
dataset = NuScencesMaps(path, split)
test = 0
