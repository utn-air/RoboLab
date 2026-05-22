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

# NEW PLOT IMPLEMENTATION
def save_goal_distance_plot(task_goal_distances, output_path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MultipleLocator
	plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})

	model_variants = []
	tasks = []
	for item in task_goal_distances:
		if item["model_variant"] not in model_variants:
			model_variants.append(item["model_variant"])
		if item["task"] not in tasks:
			tasks.append(item["task"])

	stats_by_model_and_task = {(item["model_variant"], item["task"]): item for item in task_goal_distances}

	model_positions = list(range(len(model_variants)))
	task_count = len(tasks)
	task_spacing = 0.8 / task_count
	model_variant_labels = {
		"right_vjepa": "VJEPA2",
		"right_dinov3": "DINOv3",
		"wrist_vjepa": "VJEPA2",
		"wrist_dinov3": "DINOv3",
		"ind_vjepa": "VJEPA2",
		"ind_dinov3": "DINOv3",
		"dual_vjepa": "VJEPA2",
		"dual_dinov3": "DINOv3",
		"dual_dinov3_roboarena": "VALPA",
	}
	model_variant_groups = [
		("Side", ["right_vjepa", "right_dinov3"]),
		("Wrist", ["wrist_vjepa", "wrist_dinov3"]),
		("Dual Independent", ["ind_vjepa", "ind_dinov3"]),
		("Dual Shared Latent", ["dual_vjepa", "dual_dinov3", "dual_dinov3_roboarena"]),
	]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(max(12, 0.9 * len(model_variants)), 7))
	ax = plt.gca()
	group_background_colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
	for group_index, (_, group_model_variants) in enumerate(model_variant_groups):
		group_positions = [model_variants.index(model_variant) for model_variant in group_model_variants if model_variant in model_variants]
		group_start = min(group_positions)
		group_end = max(group_positions)
		group_left = group_start - 0.5
		group_right = group_end + 0.5
		if group_index == len(model_variant_groups) - 1:
			group_right = len(model_variants) - 0.5
		ax.axvspan(group_left, group_right, color=group_background_colors[group_index], alpha=0.10, zorder=0)
	ax.set_xlim(-0.5, len(model_variants) - 0.5)

	for task_index, task in enumerate(tasks):
		means = []
		stds = []
		for model_variant in model_variants:
			stats = stats_by_model_and_task[(model_variant, task)]
			means.append(stats["goal_distance_mean"])
			stds.append(stats["goal_distance_std"])

		offset = (task_index - (task_count - 1) / 2) * task_spacing
		plot_positions = [position + offset for position in model_positions]
		label = task.replace("Reach", "").replace("Task", "")
		plt.errorbar(plot_positions, means, yerr=stds, fmt="o", capsize=3, linestyle="none", label=label)

	ax.yaxis.set_minor_locator(MultipleLocator(0.02))
	plt.grid(axis="y", which="major", linestyle="--", alpha=0.35)
	plt.grid(axis="y", which="minor", linestyle="--", alpha=0.20)
	for model_boundary in [position + 0.5 for position in model_positions[:-1]]:
		ax.axvline(model_boundary, linestyle="--", color="0.75", alpha=0.35, linewidth=0.8)
	ax.set_axisbelow(True)
	plt.ylabel("Goal Distance", labelpad=16)
	plt.xticks(model_positions, [model_variant_labels[model_variant] for model_variant in model_variants])
	plt.tick_params(axis="x", pad=6)

	for group_label, group_model_variants in model_variant_groups:
		group_positions = [model_variants.index(model_variant) for model_variant in group_model_variants if model_variant in model_variants]
		group_start = min(group_positions)
		group_end = max(group_positions)
		group_center = (group_start + group_end) / 2
		ax.text(group_center, -0.10, group_label, ha="center", va="top", fontweight="bold", transform=ax.get_xaxis_transform())

	plt.legend(ncol=2)
	plt.tight_layout()
	plt.subplots_adjust(bottom=0.18)
	plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
	plt.close()

# NEW PLOT IMPLEMENTATION
def save_goal_distance_summary_plot(model_goal_distances, output_path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MultipleLocator
	plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})

	model_variants = [item["model_variant"] for item in model_goal_distances]
	model_positions = list(range(len(model_variants)))
	goal_distance_means = [item["goal_distance_mean"] for item in model_goal_distances]
	goal_distance_stds = [item["goal_distance_std"] for item in model_goal_distances]
	model_variant_labels = {
		"right_vjepa": "VJEPA2",
		"right_dinov3": "DINOv3",
		"wrist_vjepa": "VJEPA2",
		"wrist_dinov3": "DINOv3",
		"ind_vjepa": "VJEPA2",
		"ind_dinov3": "DINOv3",
		"dual_vjepa": "VJEPA2",
		"dual_dinov3": "DINOv3",
		"dual_dinov3_roboarena": "VALPA",
	}
	model_variant_groups = [
		("Side", ["right_vjepa", "right_dinov3"]),
		("Wrist", ["wrist_vjepa", "wrist_dinov3"]),
		("Dual Independent", ["ind_vjepa", "ind_dinov3"]),
		("Dual Shared Latent", ["dual_vjepa", "dual_dinov3", "dual_dinov3_roboarena"]),
	]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(max(12, 0.9 * len(model_variants)), 7))
	ax = plt.gca()
	group_background_colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
	for group_index, (_, group_model_variants) in enumerate(model_variant_groups):
		group_positions = [model_variants.index(model_variant) for model_variant in group_model_variants if model_variant in model_variants]
		group_start = min(group_positions)
		group_end = max(group_positions)
		group_left = group_start - 0.5
		group_right = group_end + 0.5
		if group_index == len(model_variant_groups) - 1:
			group_right = len(model_variants) - 0.5
		ax.axvspan(group_left, group_right, color=group_background_colors[group_index], alpha=0.10, zorder=0)
	ax.set_xlim(-0.5, len(model_variants) - 0.5)

	plt.errorbar(model_positions, goal_distance_means, yerr=goal_distance_stds, fmt="o", capsize=5, linestyle="none", color="black", zorder=2)
	ax.yaxis.set_minor_locator(MultipleLocator(0.02))
	plt.grid(axis="y", which="major", linestyle="--", alpha=0.35)
	plt.grid(axis="y", which="minor", linestyle="--", alpha=0.20)
	for model_boundary in [position + 0.5 for position in model_positions[:-1]]:
		ax.axvline(model_boundary, linestyle="--", color="0.75", alpha=0.35, linewidth=0.8)
	ax.set_axisbelow(True)
	plt.ylabel("Goal Distance", labelpad=16)
	plt.xticks(model_positions, [model_variant_labels[model_variant] for model_variant in model_variants])
	plt.tick_params(axis="x", pad=6)

	for group_label, group_model_variants in model_variant_groups:
		group_positions = [model_variants.index(model_variant) for model_variant in group_model_variants if model_variant in model_variants]
		group_start = min(group_positions)
		group_end = max(group_positions)
		group_center = (group_start + group_end) / 2
		ax.text(group_center, -0.10, group_label, ha="center", va="top", fontweight="bold", transform=ax.get_xaxis_transform())

	plt.tight_layout()
	plt.subplots_adjust(bottom=0.18)
	plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
	plt.close()

# NEW PLOT IMPLEMENTATION
def save_steps_plot(task_steps, output_path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MultipleLocator
	plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})

	model_variants = []
	tasks = []
	for item in task_steps:
		if item["model_variant"] not in model_variants:
			model_variants.append(item["model_variant"])
		if item["task"] not in tasks:
			tasks.append(item["task"])

	stats_by_model_and_task = {(item["model_variant"], item["task"]): item for item in task_steps}

	model_positions = list(range(len(model_variants)))
	task_count = len(tasks)
	task_spacing = 0.8 / task_count
	model_variant_labels = {
		"right_vjepa": "VJEPA2",
		"right_dinov3": "DINOv3",
		"wrist_vjepa": "VJEPA2",
		"wrist_dinov3": "DINOv3",
		"ind_vjepa": "VJEPA2",
		"ind_dinov3": "DINOv3",
		"dual_vjepa": "VJEPA2",
		"dual_dinov3": "DINOv3",
		"dual_dinov3_roboarena": "VALPA",
	}
	model_variant_groups = [
		("Side", ["right_vjepa", "right_dinov3"]),
		("Wrist", ["wrist_vjepa", "wrist_dinov3"]),
		("Dual Independent", ["ind_vjepa", "ind_dinov3"]),
		("Dual Shared Latent", ["dual_vjepa", "dual_dinov3", "dual_dinov3_roboarena"]),
	]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(max(12, 0.9 * len(model_variants)), 7))
	ax = plt.gca()
	group_background_colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
	for group_index, (_, group_model_variants) in enumerate(model_variant_groups):
		group_positions = [model_variants.index(model_variant) for model_variant in group_model_variants if model_variant in model_variants]
		group_start = min(group_positions)
		group_end = max(group_positions)
		group_left = group_start - 0.5
		group_right = group_end + 0.5
		if group_index == len(model_variant_groups) - 1:
			group_right = len(model_variants) - 0.5
		ax.axvspan(group_left, group_right, color=group_background_colors[group_index], alpha=0.10, zorder=0)
	ax.set_xlim(-0.5, len(model_variants) - 0.5)

	for task_index, task in enumerate(tasks):
		means = []
		stds = []
		for model_variant in model_variants:
			stats = stats_by_model_and_task[(model_variant, task)]
			means.append(stats["steps_mean"])
			stds.append(stats["steps_std"])

		offset = (task_index - (task_count - 1) / 2) * task_spacing
		plot_positions = [position + offset for position in model_positions]
		label = task.replace("Reach", "").replace("Task", "")
		plt.errorbar(plot_positions, means, yerr=stds, fmt="o", capsize=3, linestyle="none", label=label)

	ax.yaxis.set_minor_locator(MultipleLocator(5))
	plt.grid(axis="y", which="major", linestyle="--", alpha=0.35)
	plt.grid(axis="y", which="minor", linestyle="--", alpha=0.20)
	for model_boundary in [position + 0.5 for position in model_positions[:-1]]:
		ax.axvline(model_boundary, linestyle="--", color="0.75", alpha=0.35, linewidth=0.8)
	ax.set_axisbelow(True)
	plt.ylabel("Steps", labelpad=16)
	plt.xticks(model_positions, [model_variant_labels[model_variant] for model_variant in model_variants])
	plt.tick_params(axis="x", pad=6)

	for group_label, group_model_variants in model_variant_groups:
		group_positions = [model_variants.index(model_variant) for model_variant in group_model_variants if model_variant in model_variants]
		group_start = min(group_positions)
		group_end = max(group_positions)
		group_center = (group_start + group_end) / 2
		ax.text(group_center, -0.10, group_label, ha="center", va="top", fontweight="bold", transform=ax.get_xaxis_transform())

	plt.legend(ncol=2)
	plt.tight_layout()
	plt.subplots_adjust(bottom=0.18)
	plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
	plt.close()

# NEW PLOT IMPLEMENTATION
def save_steps_summary_plot(model_steps, output_path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MultipleLocator
	plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})

	model_variants = [item["model_variant"] for item in model_steps]
	model_positions = list(range(len(model_variants)))
	steps_means = [item["steps_mean"] for item in model_steps]
	steps_stds = [item["steps_std"] for item in model_steps]
	model_variant_labels = {
		"right_vjepa": "VJEPA2",
		"right_dinov3": "DINOv3",
		"wrist_vjepa": "VJEPA2",
		"wrist_dinov3": "DINOv3",
		"ind_vjepa": "VJEPA2",
		"ind_dinov3": "DINOv3",
		"dual_vjepa": "VJEPA2",
		"dual_dinov3": "DINOv3",
		"dual_dinov3_roboarena": "VALPA",
	}
	model_variant_groups = [
		("Side", ["right_vjepa", "right_dinov3"]),
		("Wrist", ["wrist_vjepa", "wrist_dinov3"]),
		("Dual Independent", ["ind_vjepa", "ind_dinov3"]),
		("Dual Shared Latent", ["dual_vjepa", "dual_dinov3", "dual_dinov3_roboarena"]),
	]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(max(12, 0.9 * len(model_variants)), 7))
	ax = plt.gca()
	group_background_colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
	for group_index, (_, group_model_variants) in enumerate(model_variant_groups):
		group_positions = [model_variants.index(model_variant) for model_variant in group_model_variants if model_variant in model_variants]
		group_start = min(group_positions)
		group_end = max(group_positions)
		group_left = group_start - 0.5
		group_right = group_end + 0.5
		if group_index == len(model_variant_groups) - 1:
			group_right = len(model_variants) - 0.5
		ax.axvspan(group_left, group_right, color=group_background_colors[group_index], alpha=0.10, zorder=0)
	ax.set_xlim(-0.5, len(model_variants) - 0.5)

	plt.errorbar(model_positions, steps_means, yerr=steps_stds, fmt="o", capsize=5, linestyle="none", color="black", zorder=2)
	ax.yaxis.set_minor_locator(MultipleLocator(5))
	plt.grid(axis="y", which="major", linestyle="--", alpha=0.35)
	plt.grid(axis="y", which="minor", linestyle="--", alpha=0.20)
	for model_boundary in [position + 0.5 for position in model_positions[:-1]]:
		ax.axvline(model_boundary, linestyle="--", color="0.75", alpha=0.35, linewidth=0.8)
	ax.set_axisbelow(True)
	plt.ylabel("Steps", labelpad=16)
	plt.xticks(model_positions, [model_variant_labels[model_variant] for model_variant in model_variants])
	plt.tick_params(axis="x", pad=6)

	for group_label, group_model_variants in model_variant_groups:
		group_positions = [model_variants.index(model_variant) for model_variant in group_model_variants if model_variant in model_variants]
		group_start = min(group_positions)
		group_end = max(group_positions)
		group_center = (group_start + group_end) / 2
		ax.text(group_center, -0.10, group_label, ha="center", va="top", fontweight="bold", transform=ax.get_xaxis_transform())

	plt.tight_layout()
	plt.subplots_adjust(bottom=0.18)
	plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
	plt.close()


zip_paths = ["output/cleandata/right_vjepa.zip",
			"output/cleandata/right_dinov3.zip",
			 "output/cleandata/wrist_vjepa.zip",
			 "output/cleandata/wrist_dinov3.zip",
			 "output/cleandata/ind_vjepa.zip",
			 "output/cleandata/ind_dinov3.zip",
			 "output/cleandata/dual_vjepa.zip",
			 "output/cleandata/dual_dinov3.zip",
			 "output/cleandata/dual_dinov3_roboarena.zip",
			 ]

zip_paths = [Path(zip_path) for zip_path in zip_paths]
task_goal_distances = []
model_goal_distances = []
task_steps = []
model_steps = []

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

for zip_path in zip_paths:
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

		# NEW PLOT IMPLEMENTATION
		for task in tasks_statistics:
			task_goal_distances.append({
				"model_variant": zip_path.stem,
				"task": task,
				"goal_distance_mean": tasks_statistics[task]["goal_distances_mean"],
				"goal_distance_std": tasks_statistics[task]["goal_distances_std"],
			})
			task_steps.append({
				"model_variant": zip_path.stem,
				"task": task,
				"steps_mean": tasks_statistics[task]["steps_mean"],
				"steps_std": tasks_statistics[task]["steps_std"],
			})

		all_goal_distances = []
		all_steps = []
		for task in tasks_statistics:
			all_goal_distances.extend(tasks_statistics[task]["goal_distances"])
			all_steps.extend(tasks_statistics[task]["steps"])

		goal_distance_mean = sum(all_goal_distances) / len(all_goal_distances)
		goal_distance_std = (sum([(x - goal_distance_mean) ** 2 for x in all_goal_distances]) / (len(all_goal_distances) - 1)) ** 0.5
		model_goal_distances.append({
			"model_variant": zip_path.stem,
			"goal_distance_mean": goal_distance_mean,
			"goal_distance_std": goal_distance_std,
		})

		steps_mean = sum(all_steps) / len(all_steps)
		steps_std = (sum([(x - steps_mean) ** 2 for x in all_steps]) / (len(all_steps) - 1)) ** 0.5
		model_steps.append({
			"model_variant": zip_path.stem,
			"steps_mean": steps_mean,
			"steps_std": steps_std,
		})


save_goal_distance_plot(task_goal_distances, Path("output/goal_distance_by_model_and_task.png"))
print("saved goal distance plot to output/goal_distance_by_model_and_task.png")

save_goal_distance_summary_plot(model_goal_distances, Path("output/goal_distance_by_model_all_tasks.png"))
print("saved all-tasks goal distance plot to output/goal_distance_by_model_all_tasks.png")

save_steps_plot(task_steps, Path("output/steps_by_model_and_task.png"))
print("saved steps plot to output/steps_by_model_and_task.png")

save_steps_summary_plot(model_steps, Path("output/steps_by_model_all_tasks.png"))
print("saved all-tasks steps plot to output/steps_by_model_all_tasks.png")



							

			