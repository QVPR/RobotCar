import os
import argparse

import numpy as np
import pickle
from scipy.spatial.transform import Rotation
import pandas as pd

from src.settings import RAW_DIR, READY_DIR, camera_names
import src.thirdparty.robotcar_dataset_sdk as sdk
from src.thirdparty.robotcar_dataset_sdk.python.interpolate_poses import interpolate_ins_poses
from src.thirdparty.robotcar_dataset_sdk.python.transform import build_se3_transform, se3_to_components


def assign_poses(traverse, camera):
    sdk_path = os.path.abspath(sdk.__file__)
    extrinsics_dir = os.path.join(os.path.dirname(sdk_path), 'extrinsics')
    save_dir = os.path.join(READY_DIR, traverse, camera)
    if not os.path.exists(save_dir):
        os.mkdirs(save_dir)
    # retrieve list of image tstamps
    img_folder = os.path.join(READY_DIR, traverse, camera)
    img_paths = os.listdir(img_folder)
    if not img_paths:
        raise FileNotFoundError("No images ready! Run ready_images.py on traverse/camera pair first.")
    tstamps = [int(os.path.basename(img_path)[:-4]) for img_path in img_paths if img_path.endswith(".png")]
    rtk_path = os.path.join(RAW_DIR, traverse, 'rtk.csv')
    interp_poses = np.asarray(interpolate_ins_poses(rtk_path, tstamps, use_rtk=True))
    # apply camera extrinsics to INS for abs camera poses. Note extrinsics are relative
    # to the STEREO camera, so compose extrinsics for ins and mono to get true cam pose
    with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    T_ext_ste_ins = np.asarray(build_se3_transform([float(x) for x in extrinsics.split(' ')]))
    cam_name = "stereo" if "stereo" in camera else camera
    with open(os.path.join(extrinsics_dir, '{}.txt'.format(cam_name))) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    # relative pose of camera to stereo camera (origin frame)
    T_ext_stereo = np.asarray(build_se3_transform([float(x) for x in extrinsics.split(' ')]))
    # relative pose to INS/RTK sensor from stereo camera
    T_ext = np.linalg.solve(T_ext_ste_ins, T_ext_stereo)
    poses = interp_poses @ T_ext
    xyzrpys = [se3_to_components(pose) for pose in poses]
    df = pd.DataFrame(xyzrpys, columns=["northing", "easting", "down", "roll", "pitch", "yaw"])
    df.insert(0, "timestamp", tstamps, True)
    df.to_csv(os.path.join(save_dir, "camera_poses.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--traverses",
        nargs="+",
        type=str,
        required=True,
        help=(
            "<Required> Time/date stamp(s) of traverses to extract RTK poses for e.g. 2015-03-17-11-08-44."
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
            "e.g. mono_left|right|rear stereo/left|centre|right"
        ),
    )
    args = parser.parse_args() 
    cameras = camera_names if "all" in args.cameras else args.cameras
    for traverse in args.traverses:
        for camera in cameras:
            assign_poses(traverse, camera)
