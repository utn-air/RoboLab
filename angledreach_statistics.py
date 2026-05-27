from pathlib import Path
import zipfile
import h5py
import json 

folder_paths = ["output_angledreach/dual_dinov3_roboarena_angledreach",
                "output_angledreach/dual_dinov3_dual_angledreach",
                "output_angledreach/ind_dinov3_wrist_angledreach",
                "output_angledreach/right_dinov3_angledreach",
                "output_angledreach/wrist_dinov3_wrist_angledreach",
                "output_angledreach/dual_vjepa_angledreach",
                "output_angledreach/ind_vjepa_angledreach",
                "output_angledreach/right_vjepa_angledreach",
                "output_angledreach/wrist_vjepa_angledreach"
                ]


simple_tasks = [
				"AngledReachDrillTask",
				]

for folder_path in folder_paths:
    for task in simple_tasks:
        # open the run_i.hdf5 files within folder_path/task/
        for run_i in range(5):
            hdf5_path = Path(folder_path) / task / f"run_{run_i}.hdf5"
            with h5py.File(hdf5_path, "r") as f:
                print(f["data"]['demo_0']['ee_pose']['position'])
                print(f["data"]['demo_0']['ee_pose']['orientation'])
