from src.data.dataset import NuScencesMaps

path = "/Users/colinhoffmann/Desktop/nuscenes"
dataset = NuScencesMaps(path, "train_split")

for i in range(1000):
    image, calib, labels, mask = dataset[i]
    print(i)
