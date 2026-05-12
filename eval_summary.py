#!/usr/bin/env python3
"""Inspect experiment zip files and print HDF5 keys.

This script scans each .zip file under an output folder, finds task-level
`data.hdf5` and `run_*.hdf5` files, and prints their keys to help quick log
inspection and statistics setup.
"""

from pathlib import Path
import zipfile
import h5py




zip_path = "/workspace/robolab/output/dual_dinov3.zip"

zip_path = Path(zip_path)

current_task = "ReachYogurtCupTask" #"ReachCoffeeCanTask" # "ReachOrangeTask"
goal_posiiton = [0.3478515148162842, -0.17459048330783844, 0.21515005826950073]
				# [0.6726567149162292, 0.36861035227775574, 0.3434607684612274]
				# [0.5479271411895752, -0.11910949647426605, 0.24504002928733826]

with zipfile.ZipFile(zip_path, "r") as zip_file:
	for item in zip_file.infolist():

		if item.is_dir():
			# print(f"Task: {item.filename}")
			# current_task = item.filename
			continue

		if item.filename.endswith(".hdf5"):
			split_filename = item.filename.split("/")
			filename = split_filename[-1]
			
			if split_filename[1] == current_task and filename.startswith("run_"):
				print(f"File: {item.filename}")
				
				with zip_file.open(item) as file:
					with h5py.File(file, "r") as hdf5_file:
						print(f"{hdf5_file['data']['demo_0'].keys()}")
						position = hdf5_file['data']['demo_0']['ee_pose']['position'][-1]
						orientation = hdf5_file['data']['demo_0']['ee_pose']['orientation'][-1]

						# euclidean distance to goal
						distance = ((position[0] - goal_posiiton[0]) **2 + (position[1] - goal_posiiton[1]) ** 2 + (position[2] - goal_posiiton[2]) ** 2)
						print(f"Final position: {position}, distance to goal: {distance}")

						

		