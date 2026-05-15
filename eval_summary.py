#!/usr/bin/env python3
"""Inspect experiment zip files and print HDF5 keys.

This script scans each .zip file under an output folder, finds task-level
`data.hdf5` and `run_*.hdf5` files, and prints their keys to help quick log
inspection and statistics setup.
"""

from pathlib import Path
import zipfile
import h5py
import json 


zip_path = "/workspace/robolab/output/dual_dinov3.zip"

zip_path = Path(zip_path)

current_task = [
				"ReachAppleTask",
				"ReachBagelTask",
				"ReachBananaTask",
				"ReachCeramicMugTask",
				# "ReachCoffeePotTask",
				"ReachOrangeJuiceCartonTask",
				"ReachOrangeTask",
				"ReachYogurtCupTask",
				# "ReachCoffeeCanTask",
				# "ReachPitcherTask",
				# "ReachSpoonBigTask"
				]

with zipfile.ZipFile(zip_path, "r") as zip_file:
	for item in zip_file.infolist():
		# print(f"Inspecting: {item.filename}")

		if item.is_dir():
			# print(f"Task: {item.filename}")
			# current_task = item.filename
			continue

		if item.filename.endswith(".hdf5"):
			split_filename = item.filename.split("/")
			filename = split_filename[-1]
			
			if split_filename[1] in current_task and filename.startswith("run_"):
				# print(f"File: {item.filename}")

				# read "/assets/wm_tasks/{split_filename[1]}/status.json" to get goal position
				status_file = Path(f"assets/wm_tasks/{split_filename[1]}/status.json")
				with status_file.open("r", encoding="utf-8") as handle:
					status_data = json.load(handle)
				goal_posiiton = status_data["last_ee_pose"][:3]
				
				with zip_file.open(item) as file:
					with h5py.File(file, "r") as hdf5_file:
						# print(f"{hdf5_file['data']['demo_0'].keys()}")
						position = hdf5_file['data']['demo_0']['ee_pose']['position']
						orientation = hdf5_file['data']['demo_0']['ee_pose']['orientation']

						# calculate number of steps in the run
						for i in range(position.shape[0]):
							current_position = position[i]
							distance = ((current_position[0] - goal_posiiton[0]) **2 + (current_position[1] - goal_posiiton[1]) ** 2 + (current_position[2] - goal_posiiton[2]) ** 2) ** 0.5
							if distance < 0.05 
								print(f"file {item.filename}: reached goal at step {i}, distance: {distance}")
								break

					# # compute distance to the goal from the last position in the run
					# distance = ((position[-1][0] - goal_posiiton[0]) **2 + (position[-1][1] - goal_posiiton[1]) ** 2 + (position[-1][2] - goal_posiiton[2]) ** 2) ** 0.5
					# print(f"file {item.filename}: distance to goal: {distance}")




						

		