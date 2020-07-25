import os
import argparse
import shutil

import numpy as np
import pandas as pd

from src.settings import RAW_DIR, READY_DIR, camera_names
from src.util import geometry
from src.util.geometry import SE3


def create_traverse(traverse, camera, w, thres):
    # import camera poses
    img_path = os.path.join(READY_DIR, traverse, camera)
    df = pd.read_csv(os.path.join(img_path, "camera_poses.csv"))
    xyzrpy = df[["northing", "easting", "down", "roll", "pitch", "yaw"]].to_numpy()
    poses = SE3.from_xyzrpy(xyzrpy)
    # generate subsampled traverse
    indices = [0] # first image in set of keyframes
    pose_curr = poses[0]
    for i in range(1, len(poses)):
        curr_diff = geometry.metric(pose_curr, poses[i], w)
        if curr_diff > thres:
            indices.append(i)
            pose_curr = poses[i]
    indices = np.asarray(indices)
    subsampled = df.iloc[indices]
    return subsampled


def save_traverse(path, df, traverse, camera):
    # copy over all selected images to new path
    ss_tstamps = np.squeeze(df[["timestamp"]].to_numpy())
    img_paths = [os.path.join(READY_DIR, traverse, camera, str(tstamp) + ".png") for tstamp in ss_tstamps]
    save_path = os.path.join(path, camera, "images")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for img_path in img_paths:
        shutil.copy2(img_path, os.path.join(save_path))
    df.to_csv(os.path.join(save_path, "..", "camera_poses.csv"))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample full traverse at regular spatial intervals")
    parser.add_argument(
        "-t",
        "--traverse",
        type=str,
        required=True,
        help=(
            "<Required> Time/date stamp of traverses to subsample e.g. 2015-03-17-11-08-44."
        ),
    )
    parser.add_argument(
        "-c",
        "--cameras",
        nargs="+",
        type=str,
        required=True,
        help=(
            "<Required> Cameras to extract poses for within a traverse"
            "e.g. mono_(left|right|rear) stereo/(left|centre|right)"
        ),
    )
    parser.add_argument(
        "-w",
        "--attitude-weight",
        type=float,
        default=10,
        help=(
            "weight for attitude components d where 1 / d rad rotation is equivalent to 1m translation"
        ),
    )
    parser.add_argument(
       "-d",
       "--distance",
       type=float,
       default=2,
       help=(
           "distance threshold on weighted pose distance to insert new image"
       )
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help=(
            "<Required> path for traverse folder"
        ),
    )
    args = parser.parse_args()

    cameras = camera_names if "all" in args.cameras else args.cameras
    for camera in cameras:
        df = create_traverse(args.traverse, camera, args.attitude_weight, args.distance)
        save_traverse(args.path, df, args.traverse, camera)
