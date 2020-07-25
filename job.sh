#!/bin/bash -l
#PBS -N ReadyImage
#PBS -l select=1:ncpus=8:mem=4000mb,walltime=3:59:01
cd /work/qvpr/workspace/RobotCar
conda activate ./envs
python3 ready_images.py -t 2015-03-24-13-47-33 -c stereo/left -n 8
