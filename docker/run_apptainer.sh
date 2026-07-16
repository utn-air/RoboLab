
apptainer exec --nv \
  --overlay ../robolab-overlay.img \
  ../robolab.sif \
  python-rtx-compat examples/run_empty.py \
  --headless \
  --task MustardInLeftBinTask \
  --num-steps 5 \
  > ../sweep_vulkan_debug.log 2>&1