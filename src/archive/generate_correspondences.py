import os
import sys
import argparse
import csv

import numpy as np
import pickle

from src.util import geometry

processed_path = os.path.abspath("/work/qvpr/data/processed/RobotCar/")
csv_path = "/work/qvpr/workspace/RobotCar/examples/"

def build_reference_keyframes(gt, threshold, attitude_weight):
    """
    Generates set of indices corresponding to image timestamps for
    a traverse, where each timestamp corresponds to a camera pose that 
    has regular spatial separation (threshold) from the previous entry.
    """
    indices = [0] # first image in set of keyframes
    gt_curr = gt[0]
    for i in range(1, len(gt)):
        curr_diff = geometry.metric(gt_curr, gt[i], attitude_weight)
        if curr_diff > threshold:
            indices.append(i)
            gt_curr = gt[i]
    indices = np.asarray(indices)
    return indices

# for each query, cycle through reference and extract nearest query frame in pose
def correspondences(ref, query, attitude_weight):
    ind_q = []
    for T in ref:
        d = geometry.metric(T, query, attitude_weight)
        #print(T.t(), query[np.argmin(d)].t())
        #print((T.R().inv() *  query[np.argmin(d)].R()).magnitude())
        ind_q.append(np.argmin(d))
    return np.asarray(ind_q)

if __name__ == "__main__":
    """
    TO DO: apply for arbitrary cameras.
    """
    parser = argparse.ArgumentParser(description="Generate 1-to-1 correspondences between reference and query traverses")
    parser.add_argument('-r', '--reference', type=str, required=True,
                    help="<Required> name of reference traverse to generate correspondences to.")
    parser.add_argument('-q', '--query', nargs='+', type=str, required=True,
                    help="<Required> query traverses to retrieve correspondences from.")
    parser.add_argument('-w', '--attitude-weight', type=float, default=0, 
        help="weight for attitude components of pose distance equal to d where 1 / d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-k', '--kf-threshold', type=float, default=1, help="threshold on weighted pose distance to generate new keyframe")
    args = parser.parse_args()
    
    ref_rtk_dir = os.path.join(processed_path, args.reference, "rtk")
    with open(ref_rtk_dir + '/SE3_stereo_left.pickle', 'rb') as f:
        ref_SE3 = pickle.load(f)
    ref_tstamps = np.load(ref_rtk_dir + '/tstamps_stereo_left.npy')            
    indices = build_reference_keyframes(ref_SE3, args.kf_threshold, args.attitude_weight)
    ref_tstamps_sub = ref_tstamps[indices]

    all_tstamps = [ref_tstamps_sub]
    for name in args.query:
        query_rtk_dir = os.path.join(processed_path, name, "rtk")
        with open(query_rtk_dir + '/SE3_stereo_left.pickle', 'rb') as f:
            query_SE3 = pickle.load(f)    
        query_tstamps = np.load(query_rtk_dir + '/tstamps_stereo_left.npy')            
        ind_q = correspondences(ref_SE3[indices], query_SE3, args.attitude_weight)
        all_tstamps.append(query_tstamps[ind_q])
    all_tstamps = np.asarray(all_tstamps).transpose()

    with open(csv_path + '/correspondences.csv', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow([args.reference] + args.query)
        for row in all_tstamps:
            writer.writerow(row)
