# RobotCar dataset

This repo contains code to process the RobotCar dataset.

# Use on HPC

Go into the repo directory on HPC `/work/qvpr/workspace/RobotCar/` and run `conda activate ./envs` to activate the conda environment with required dependencies to run the scripts.

# Local machine use

## Initial setup

Assuming your pwd is in the base directory of this repo, first use `pip install -e .` to install the repo as a package.

In addition, go to `src/settings.py` and change the `RAW_DIR` variable to be the base directory for the raw RobotCar dataset on your local machine. Download the raw data from HPC at path `/work/qvpr/data/raw/RobotCar/` using rsync or however you want.

To setup the conda environment will all dependencies, use `conda create --name RobotCar --file requirements.txt` in the base directory of the repo.

## Ready images

This codebase allows you to undistort and deBayerize the raw images using the `src/process_raw/ready_images.py` script and saves them to the path specified under `READY_DIR` in `src/settings.py`.

## Interpolate RTK poses to images

This codebase allows you to interpolate the RTK poses to the camera image timestamps for both stereo and monocular cameras. The camera extrinsics relative to the GPS is given, providing poses in a globally consistent coordinate frame for all images. See the `src/process_raw/gps_camera_align.py` script more more information around usage. The output of the poses will be in the file `$READY_DIR/traverse_name/camera_name/camera_poses.csv` in xyzrpy format.

# Useful tips

The `src/utils/geometry.py` file contains some useful functions and classes for representing and performing operations with 6DOF transformations that will come in handy for finding image correspondences.
