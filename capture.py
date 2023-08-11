import sys
import os
import time
import datetime
from multiprocessing import Process, Value
from multiprocessing.sharedctypes import Synchronized
import cv2
from ctypes import c_bool


def main(cam, name, write: Synchronized, stop: Synchronized):
    """
    Script to capture plugged in USB webcams (Logitech C920e).

    Ensure that the USB Bandwith is sufficient.

    Specify the camera to capture by the index.
    Which index refers to which camera depends on the local machine.

    Args:
        cam (int): Camera ID
        name (str): Camera name
        write (c_bool): Flag for capturing start
        stop (c_bool): Flag for capturing stop

    Returns:
        None

    Raises:
        None
    """

    path_name = f"data/real/{name}"
    if not os.path.exists(path_name):
        os.mkdir(path_name)
        os.mkdir(path_name + "/videos")
        os.mkdir(path_name + "/timestamps")

    file = open(f"{path_name}/timestamps/CAM{cam}.txt", "w")

    cap = cv2.VideoCapture(cam, cv2.CAP_AVFOUNDATION)
    cap.set(3, 1920)
    cap.set(4, 1080)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fps = 1

    name = f"data/real/{name}/videos/CAM{cam}.avi"
    writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    while True:
        key = None
        tik = time.time()
        ret, frame = cap.read()

        if ret and write.value:
            writer.write(frame)
            file.write(str(datetime.datetime.now()) + "\n")

        if ret:
            cv2.imshow(f"CAM{cam}", frame)

            tok = time.time()
            elapsed = tok - tik
            delay = max(int((1 / fps - elapsed) * 1000), 1)
            key = cv2.waitKey(delay) & 0xFF

        if key == ord("1"):
            write.value = True
            print("start")
        if key == ord("2"):
            write.value = False
            print("stop")
        if key == ord("3"):
            stop.value = True

        if stop.value:
            break

    cap.release()
    writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    name = sys.argv[1]
    cams = sys.argv[2:]
    write = Value(c_bool, False)
    stop = Value(c_bool, False)
    for cam in cams:
        Process(target=main, args=(int(cam), name, write, stop)).start()
        time.sleep(5)
