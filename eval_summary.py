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


zip_path = "output/cleandata/dual_dinov3_roboarena.zip"

zip_path = Path(zip_path)

simple_tasks = [
				"ReachAppleTask",
				"ReachBagelTask",
				"ReachBananaTask",
				"ReachCeramicMugTask",
				# "ReachCoffeePotTask",
				"ReachOrangeJuiceCartonTask",
				"ReachOrangeTask",
				"ReachYogurtCupTask",
				]
edge_tasks = [	
			"ReachCoffeeCanTask",
			"ReachPitcherTask",
			"ReachSpoonBigTask"
			]

with zipfile.ZipFile(zip_path, "r") as zip_file:
	tasks_statistics = {task: {
								"total_runs": 0,
								"steps": [], 
								"steps_mean": 0, 
								"steps_std": 0, 
								"goal_distances": [],
								"goal_distances_mean": 0,
								"goal_distances_std": 0,
								"path_length": [],
								"path_length_mean": 0,
								"path_length_std": 0,
								"successful_runs": 0,
								"success_rate": 0,
								} for task in simple_tasks}
	# update tasks_statistics with edge_tasks
	for task in edge_tasks:
		tasks_statistics[task] = {
								"total_runs": 0,
								"steps": [], 
								"steps_mean": 0, 
								"steps_std": 0, 
								"goal_distances": [],
								"goal_distances_mean": 0,
								"goal_distances_std": 0,
								"path_length": [],
								"path_length_mean": 0,
								"path_length_std": 0,
								"successful_runs": 0,
								"success_rate": 0,
								}

	for item in zip_file.infolist():
		# print(f"Inspecting: {item.filename}")

		if item.is_dir():
			# print(f"Task: {item.filename}")
			# current_task = item.filename
			continue

		if item.filename.endswith(".hdf5"):
			split_filename = item.filename.split("/")
			filename = split_filename[-1]
			
			if (split_filename[1] in simple_tasks or split_filename[1] in edge_tasks) and filename.startswith("run_"):
				# print(f"File: {item.filename}")

				# read "/assets/wm_tasks/{split_filename[1]}/status.json" to get goal position
				status_file = Path(f"assets/wm_tasks/{split_filename[1]}/status.json")
				with status_file.open("r", encoding="utf-8") as handle:
					status_data = json.load(handle)
				goal_posiiton = status_data["last_ee_pose"][:3]
				
				with zip_file.open(item) as file:
					with h5py.File(file, "r") as hdf5_file:
						# print(f"{hdf5_file['data']['demo_0'].keys()}")

						tasks_statistics[split_filename[1]]["total_runs"] += 1

						position = hdf5_file['data']['demo_0']['ee_pose']['position']
						orientation = hdf5_file['data']['demo_0']['ee_pose']['orientation']

						path_length = 0
						successful_run = False
						for i in range(position.shape[0]):
							current_position = position[i, :]
							distance = ((current_position[0] - goal_posiiton[0]) **2 + (current_position[1] - goal_posiiton[1]) ** 2 + (current_position[2] - goal_posiiton[2]) ** 2) ** 0.5

							# path length
							if i > 0:
								prev_position = position[i-1]
								path_length += ((current_position[0] - prev_position[0]) **2 + (current_position[1] - prev_position[1]) ** 2 + (current_position[2] - prev_position[2]) ** 2) ** 0.5

							if split_filename[1] in simple_tasks:
								if distance <= 0.05:
									successful_run = True
									tasks_statistics[split_filename[1]]["steps"].append(i)
									tasks_statistics[split_filename[1]]["goal_distances"].append(distance)
									tasks_statistics[split_filename[1]]["path_length"].append(path_length)
									tasks_statistics[split_filename[1]]["successful_runs"] += 1
									print(f"file {item.filename}: reached goal at step {i}, distance: {distance}")

									break

							elif split_filename[1] in edge_tasks:
								if distance <= 0.1:
									successful_run = True
									tasks_statistics[split_filename[1]]["steps"].append(i)
									tasks_statistics[split_filename[1]]["goal_distances"].append(distance)
									tasks_statistics[split_filename[1]]["path_length"].append(path_length)
									tasks_statistics[split_filename[1]]["successful_runs"] += 1
									print(f"file {item.filename}: reached goal at step {i}, distance: {distance}")
									break


						# compute distance to the goal from the last position in the run
						# print(f"file {item.filename}: distance to goal: {distance}")
						if not successful_run:
							last_position = position[-1]
							distance = ((last_position[0] - goal_posiiton[0]) **2 + (last_position[1] - goal_posiiton[1]) ** 2 + (last_position[2] - goal_posiiton[2]) ** 2) ** 0.5
							tasks_statistics[split_filename[1]]["steps"].append(position.shape[0])
							tasks_statistics[split_filename[1]]["goal_distances"].append(distance)
							tasks_statistics[split_filename[1]]["path_length"].append(path_length)
							print(f"file {item.filename}: did not reach goal, final distance: {distance}")	

	# compute success rate and mean, std of steps and goal distances for each task
	for task in tasks_statistics:

		tasks_statistics[task]["steps_mean"] = sum(tasks_statistics[task]["steps"]) / len(tasks_statistics[task]["steps"])
		tasks_statistics[task]["steps_std"] = (sum([(x - tasks_statistics[task]["steps_mean"]) ** 2 for x in tasks_statistics[task]["steps"]]) / (len(tasks_statistics[task]["steps"]) - 1)) ** 0.5

		tasks_statistics[task]["goal_distances_mean"] = sum(tasks_statistics[task]["goal_distances"]) / len(tasks_statistics[task]["goal_distances"])
		tasks_statistics[task]["goal_distances_std"] = (sum([(x - tasks_statistics[task]["goal_distances_mean"]) ** 2 for x in tasks_statistics[task]["goal_distances"]]) / (len(tasks_statistics[task]["goal_distances"]) - 1)) ** 0.5

		tasks_statistics[task]["path_length_mean"] = sum(tasks_statistics[task]["path_length"]) / len(tasks_statistics[task]["path_length"])
		tasks_statistics[task]["path_length_std"] = (sum([(x - tasks_statistics[task]["path_length_mean"]) ** 2 for x in tasks_statistics[task]["path_length"]]) / (len(tasks_statistics[task]["path_length"]) - 1)) ** 0.5
		
		tasks_statistics[task]["success_rate"] = tasks_statistics[task]["successful_runs"] / tasks_statistics[task]["total_runs"]

	print(
		f"{'Task':<32} {'Runs':>4} {'Succ':>4} {'SR':>6} "
		f"{'Step Mean':>10} {'Step Std':>9} "
		f"{'Path Mean':>10} {'Path Std':>9} "
		f"{'Dist Mean':>10} {'Dist Std':>9}"
	)
	print("-" * 122)
	for task in tasks_statistics:
		stats = tasks_statistics[task]
		print(
			f"{task:<32} {stats['total_runs']:>4} {stats['successful_runs']:>4} {stats['success_rate']:>6.2f} "
			f"{stats['steps_mean']:>10.2f} {stats['steps_std']:>9.2f} "
			f"{stats['path_length_mean']:>10.4f} {stats['path_length_std']:>9.4f} "
			f"{stats['goal_distances_mean']:>10.4f} {stats['goal_distances_std']:>9.4f}"
		)

	print(f"total success rate: {sum([tasks_statistics[task]['successful_runs'] for task in tasks_statistics]) / sum([tasks_statistics[task]['total_runs'] for task in tasks_statistics]):.2f}")



						

		