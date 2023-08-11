import json
from tqdm import tqdm
import cv2
from PIL import Image


def main():
    file = open("data/real/metadata.json")
    samples = json.load(file)
    tokens = [sample["drone_token"] for sample in samples]

    file = open("data/real/drone/metadata_drone.json")
    tokens_drone = json.load(file)
    tokens_drone = [t["token"] for t in tokens_drone]

    drone = cv2.VideoCapture("data/real/drone/drone.mp4")

    for token in tqdm(tokens, total=len(tokens)):
        i = tokens_drone.index(token)
        drone.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame).save(f"data/real/drone/frames/{token}.png")


if __name__ == "__main__":
    main()
