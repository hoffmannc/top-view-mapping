import sys
from datetime import datetime
import cv2
import bisect


def setup_drone():
    drone_capture = cv2.VideoCapture("data/real/drone/drone.mp4")
    drone_capture.set(3, 1920)
    drone_capture.set(3, 1080)
    n_frames_drone = drone_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_drone = drone_capture.get(cv2.CAP_PROP_FPS)
    drone_timestamps = [
        1 / fps_drone * stamp for stamp in list(range(int(n_frames_drone)))
    ]
    return drone_capture, drone_timestamps


def setup_onboard(cam):
    onboard_capture = cv2.VideoCapture(f"data/real/camera/videos/CAM{cam}.mp4")

    timestamps = (
        open(f"data/real/camera/timestamps/CAM{cam}.txt", "r").read().split("\n")
    )
    timestamps = [timestamp for timestamp in timestamps if timestamp]
    timestamps = [
        datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f") for timestamp in timestamps
    ]
    reference = timestamps[0]
    deltas = [timestamp - reference for timestamp in timestamps]
    onboard_timestamps = [delta.total_seconds() for delta in deltas]
    return onboard_capture, onboard_timestamps


def find(name, capture, timestamps, start):
    """
    Show video footage frame and corresponding frame ID on screen.

    Store the data in the root.
    Run this script in two termonals for both footages to be synchronized.
    Find two corresponding frames.

    Args:
        name (str): camX or drone
        capture (cv2.VideoCapture): Video capture object
        timestamps (list): Corresponding timestamps to footage
        start (int): Index of frame at the start

    Returns:
        None

    Raises:
        None
    """
    index = bisect.bisect_left(timestamps, start)
    while True:
        capture.set(1, index)
        _, frame = capture.read()
        cv2.putText(
            frame,
            str(index),
            (200, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.imshow(f"{name}", frame)

        key = cv2.waitKey(0) & 0xFF

        if key == 3:
            index += 1
        if key == 2:
            index -= 1
        if key == ord("q"):
            break


if __name__ == "__main__":
    mode = sys.argv[1]
    start = int(sys.argv[2])
    if mode == "drone":
        capture, timestamps = setup_drone()
        find("Drone", capture, timestamps, start)
    if mode[:3] == "cam":
        cam = int(mode[-1])
        capture, timestamps = setup_onboard(cam)
        factor = 6
        find(f"CAM{cam}", capture, timestamps, start * factor)
