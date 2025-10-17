#!/bin/bash

LOGFILE="graphdit_tmux_$(date +%Y%m%d_%H%M%S).log"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "------------------------------------------------" | tee -a $LOGFILE
echo "GraphDiT Job Started: $(date)" | tee -a $LOGFILE
echo "Running on host: $(hostname)" | tee -a $LOGFILE
echo "Working directory: $(pwd)" | tee -a $LOGFILE
echo "------------------------------------------------" | tee -a $LOGFILE

# Properly initialize Conda in non-interactive shell
source /m2_4tdata/rzhang2/miniconda3/etc/profile.d/conda.sh

#conda init bash
conda activate phydit

# Use only GPU #0
export CUDA_VISIBLE_DEVICES=3

echo "Python path: $(which python)" | tee -a $LOGFILE
echo "Conda env: $(conda info --envs | grep \* | awk '{print $1}')" | tee -a $LOGFILE

# Generate evaluated tasks and related properties
echo "Generate .yaml file for inputs..." | tee -a $LOGFILE
# Notice properties used in evaluation could be different from properties used in guidance
python generate_registry.py \
  --task_dict '{"BG":["BG", "SA", "SC"]}' \
  --task_types '{"BG":["regression", "regression", "regression"]}' | tee -a $LOGFILE

# Run the script
echo "Starting Python script..." | tee -a $LOGFILE

python main.py --config-name=config.yaml \
    model.ensure_connected=True \
    dataset.task_name='BG' \
    dataset.guidance_target='BG-SA-SC' | tee -a $LOGFILE

echo "------------------------------------------------" | tee -a $LOGFILE
echo "GraphDiT Job Finished: $(date)" | tee -a $LOGFILE
echo "------------------------------------------------" | tee -a $LOGFILE


# python generate_registry.py \
#   --task_dict '{"TC_BM":["TC", "BM", "SA", "SC"]}' \
#   --task_types '{"TC_BM":["regression", "regression", "regression", "regression"]}' | tee -a $LOGFILE

# # Run the script
# echo "Starting Python script..." | tee -a $LOGFILE

# python main.py --config-name=config.yaml \
#     model.ensure_connected=True \
#     dataset.task_name='TC_BM' \
#     dataset.guidance_target='TC-BM-SA-SC' | tee -a $LOGFILE

# echo "------------------------------------------------" | tee -a $LOGFILE
# echo "GraphDiT Job Finished: $(date)" | tee -a $LOGFILE
# echo "------------------------------------------------" | tee -a $LOGFILE
