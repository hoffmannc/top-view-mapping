import os
import random

files = os.listdir("data/unity/samples")

samples = [file.split(".")[0] for file in files]

random.shuffle(samples)

split_value = int(len(samples) * 0.8)

train_samples = samples[:split_value]
val_samples = samples[split_value:]

with open("data/unity/splits/train_split.txt", "w") as f:
    f.write("\n".join(train_samples))

with open("data/unity/splits/val_split.txt", "w") as f:
    f.write("\n".join(val_samples))
