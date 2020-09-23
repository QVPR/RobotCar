import os
import argparse
import sys
import itertools
from multiprocessing import Pool

from tqdm import tqdm, trange
import PIL

from src.settings import RAW_DIR, READY_DIR
import src.thirdparty.robotcar_dataset_sdk as sdk
from src.thirdparty.robotcar_dataset_sdk.python import image
from src.thirdparty.robotcar_dataset_sdk.python.camera_model import CameraModel
sdk_dir = os.path.dirname(sdk.__file__)
models_dir = os.path.join(sdk_dir, "models")


def validate_images(traverse, camera):
	"""
	TODO:
	Check set of images against timestamps file to identify any missing images.
	"""
	return None


def process_and_save_image(paths):
	img_path = paths[0]
	save_path = paths[1]
	cm = paths[2]
	arr = image.load_image(img_path, cm)
	img = PIL.Image.fromarray(arr)
	img.save(save_path)
	return


def ready_images(traverse, camera, nWorkers=4, overwrite=True):
	# load and undistort images
	image_folder_path = os.path.join(RAW_DIR, traverse, camera)
	cm = CameraModel(
		models_dir, image_folder_path
		)  # this is a variable used in process_and_save_image
	try:
		fnames = os.listdir(image_folder_path)
	except FileNotFoundError:
		print("Folder {} not found, please check that this traverse/camera combination exists.")
		return
	ready_folder_path = os.path.join(READY_DIR, traverse, camera, "images")
	if not overwrite and os.path.exists(ready_folder_path):
		ready_fnames = os.listdir(ready_folder_path)
		img_fnames = [img_path for img_path in os.listdir(image_folder_path) 
			if img_path.endswith(".png") and img_path not in ready_fnames]
	else:
		img_fnames = [img_path for img_path in os.listdir(image_folder_path) if img_path.endswith(".png")]
		if not os.path.exists(ready_folder_path):
			os.makedirs(ready_folder_path)
	try:
		full_raw_path = [os.path.join(image_folder_path, fname) for fname in img_fnames]
		full_ready_path = [os.path.join(ready_folder_path, fname) for fname in img_fnames]
		with Pool(nWorkers) as pool:
			ready_images = list(tqdm(pool.imap(process_and_save_image,
								zip(full_raw_path, full_ready_path, itertools.repeat(cm))),
								total=len(full_raw_path)))
	except FileNotFoundError as e:
		print(e)
	return
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Undistorts images from the RobotCar dataset")
	parser.add_argument('-t', '--traverses', nargs='+', type=str, required=True,
			help="<Required> traverses to undistort, e.g. 2014-11-21-16-07-03 2015-03-17-11-08-44."
			    "Input 'all' instead to process all available raw traverses.")
	parser.add_argument('-c', '--cameras', nargs='+', type=str, required=True,
			choices = ["stereo/left", "stereo/right", "stereo/centre", "mono_left", "mono_right", "mono_rear"],
			help="<Required> which cameras to undistort for each traverse. Valid options include"
			"stereo/left, stereo/right, stereo/centre, mono_left, mono_right, mono_rear")
	parser.add_argument('-n', '--nWorkers', type=int, default=8, help="Number of workers to use")
	parser.add_argument('-o', '--overwrite', action='store_true',
			help="Overwrite already completed images i.e. do not resume.")
	args = parser.parse_args()
	
	# parse traverse directories
	all_traverses = [f for f in os.listdir(RAW_DIR) if f.startswith("201")]
	if 'all' in args.traverses:	
		traverses = all_traverses
	else:
		traverses = args.traverses

	for i in trange(len(traverses)):
		for j in trange(len(args.cameras)):
			ready_images(traverses[i], args.cameras[j], args.nWorkers, args.overwrite)
			tqdm.write("traverse {} and camera {} complete!".format(traverses[i], args.cameras[j]))
