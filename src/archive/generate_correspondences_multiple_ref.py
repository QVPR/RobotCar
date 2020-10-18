import os
import sys
import argparse
import csv
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from src.util import geometry

processed_path = os.path.abspath("/work/qvpr/data/ready/RobotCar/")
csv_path = "/work/qvpr/workspace/RobotCar/"

def build_reference_keyframes(gt, threshold, attitude_weight):
    """
    Generates set of indices corresponding to image timestamps for
    a traverse, where each timestamp corresponds to a camera pose that 
    has regular spatial separation (threshold) from the previous entry.
    """
    indices = [0] # first image in set of keyframes
    gt_curr = gt[0]
    for i in tqdm(range(1, len(gt))):
        curr_diff = geometry.metric(gt_curr, gt[i], attitude_weight)
        if curr_diff > threshold:
            indices.append(i)
            gt_curr = gt[i]
    indices = np.asarray(indices)
    return indices

# for each query, cycle through reference and extract nearest query frame in pose
def correspondences(ref, query, attitude_weight):
    ind_q = []
    for T in tqdm(ref):
        d = geometry.metric(T, query, attitude_weight)
        best_match = np.argmin(d)
        if d[best_match] > 10:
            tqdm.write(str(d[best_match]))
        #print(T.t(), query[np.argmin(d)].t())
        #print((T.R().inv() *  query[np.argmin(d)].R()).magnitude())
        ind_q.append(best_match)
    return np.asarray(ind_q)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 1-to-1 correspondences between reference and query traverses")
    parser.add_argument('-t', '--traverses', nargs='+', type=str, required=True,
                    help="<Required> name of traverses to generate correspondences to.")
    parser.add_argument('-b', '--baseref', type=str, required=True,
                    help="<Required> base traverse to subsample etc.")
    parser.add_argument('-w', '--attitude-weight', type=float, default=15, 
        help="weight for attitude components of pose distance equal to d where 1 / d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-k', '--kf-threshold', type=float, default=1, help="threshold on weighted pose distance to generate new keyframe")
    args = parser.parse_args()

    base_camera_poses = pd.read_csv(os.path.join(processed_path, args.baseref, 'stereo', 'left', 'camera_poses.csv'), header=0)
    base_coords = base_camera_poses.iloc[:, 2:].values
    base_SE3 = geometry.SE3.from_xyzrpy(base_coords)
    base_tstamps = base_camera_poses.iloc[:, 1].values

    indices = build_reference_keyframes(base_SE3, args.kf_threshold, args.attitude_weight)
    base_tstamps_sub = base_tstamps[indices]
    base_SE3 = base_SE3[indices]

    all_tstamps = []
    for name in args.traverses:
        traverse_camera_poses = pd.read_csv(os.path.join(processed_path, name, 'stereo', 'left', 'camera_poses.csv'), header=0)
        traverse_coords = traverse_camera_poses.iloc[:, 2:].values
        traverse_SE3 = geometry.SE3.from_xyzrpy(traverse_coords)
        traverse_tstamps = traverse_camera_poses.iloc[:, 1].values

        ind_t = correspondences(base_SE3, traverse_SE3, args.attitude_weight)
        all_tstamps.append(traverse_tstamps[ind_t])
    all_tstamps = np.asarray(all_tstamps).transpose()

    with open(csv_path + '/correspondences_multiple_ref.csv', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(args.traverses)
        for row in all_tstamps:
            writer.writerow(row)

