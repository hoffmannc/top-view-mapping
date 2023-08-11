import json
import cv2
from PIL import Image
from tqdm import tqdm


def main():
    """
    Creates single frames from all onboard camera footage.

    Store the data in the root.
    Run create_real_metadata.py before.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    file = open("data/real/metadata.json")
    samples = json.load(file)
    tokens = [sample["token"] for sample in samples]

    cams = [0, 1, 2, 4]
    for cam in cams:
        capture = cv2.VideoCapture(f"data/real/camera/videos/CAM{cam}.mp4")

        file = open(f"data/real/camera/meta/metadata_CAM{cam}.json")
        tokens_camera = [t["token"] for t in json.load(file)]

        for token in tqdm(tokens):
            if token in tokens_camera:
                i = tokens_camera.index(token)
                capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame).save(f"data/real/samples/{token}.png")


if __name__ == "__main__":
    main()
