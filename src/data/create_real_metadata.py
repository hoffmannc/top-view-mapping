import json
import cv2
import bisect
from datetime import datetime
import secrets


def map_list_elements(A, B):
    mappings = []
    for i, a in enumerate(A):
        index = bisect.bisect_left(B, a)
        if index == 0:
            continue
            # mappings.append((i, index))
        elif index == len(B):
            continue
            # mappings.append((i, index))
        else:
            diff_after = B[index] - a
            diff_before = a - B[index - 1]
            if diff_after > diff_before:
                mappings.append((i, index - 1))
            else:
                mappings.append((i, index))
    return mappings


def main():
    """
    Create metadata for real-world dataset.

    Store the data in the root.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    drone = cv2.VideoCapture("data/real/drone/drone.mp4")
    n_frames_drone = drone.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_drone = drone.get(cv2.CAP_PROP_FPS)
    drone.release()
    timestamps_drone = [
        1 / fps_drone * stamp for stamp in list(range(int(n_frames_drone)))
    ]
    tokens_drone = [secrets.token_hex(nbytes=10) for t in timestamps_drone]
    assert len(set(tokens_drone)) == len(tokens_drone)

    meta_drone = []
    for i, (token, timestamp) in enumerate(zip(tokens_drone, timestamps_drone)):
        sample = {
            "id": i,
            "timestamp": timestamp,
            "token": token,
        }
        meta_drone.append(sample)
    with open("data/real/drone/metadata_drone.json", "w") as outfile:
        json.dump(meta_drone, outfile)

    cams = [2, 0, 1, 4]
    views = ["LEFT", "MIDLEFT", "MIDRIGHT", "RIGHT"]
    syncs = [(1511, 6819), (1471, 6578), (1064, 4133), (1062, 4133)]

    metadata = []

    for cam, view, sync in zip(cams, views, syncs):
        timestamps_camera = (
            open(f"data/real/camera/timestamps/CAM{cam}.txt", "r").read().split("\n")
        )
        timestamps_camera = [t for t in timestamps_camera if t]
        timestamps_camera = [
            datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in timestamps_camera
        ]
        deltas = [t - timestamps_camera[0] for t in timestamps_camera]
        timestamps_camera = [delta.total_seconds() for delta in deltas]

        tokens_camera = [secrets.token_hex(nbytes=10) for t in timestamps_camera]
        assert len(set(tokens_camera)) == len(tokens_camera)

        meta_camera = []
        for i, (token, timestamp) in enumerate(zip(tokens_camera, timestamps_camera)):
            sample = {
                "id": i,
                "timestamp": timestamp,
                "token": token,
            }
            meta_camera.append(sample)
        with open(f"data/real/camera/meta/metadata_CAM{cam}.json", "w") as outfile:
            json.dump(meta_camera, outfile)

        synched_cam_timestamps = [
            t - timestamps_camera[sync[0]] for t in timestamps_camera
        ]
        synched_drone_timestamps = [
            t - timestamps_drone[sync[1]] for t in timestamps_drone
        ]

        mappings = map_list_elements(synched_cam_timestamps, synched_drone_timestamps)

        for i, mapping in enumerate(mappings):
            i_cam, i_drone = mapping
            sample = {
                "token": tokens_camera[i_cam],
                "cam_frame_id": i_cam,
                "cam_timestamp": timestamps_camera[i_cam],
                "cam_view": view,
                "drone_frame_id": i_drone,
                "drone_timestamp": timestamps_drone[i_drone],
                "drone_token": tokens_drone[i_drone],
            }
            metadata.append(sample)

    with open(f"data/real/metadata.json", "w") as outfile:
        json.dump(metadata, outfile)


if __name__ == "__main__":
    main()
