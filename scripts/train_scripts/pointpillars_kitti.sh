#!/bin/bash
#SBATCH -p gpu 
#SBATCH -c 4 
#SBATCH --gres=gpu:1 

# if [ "$#" -ne 2 ]; then
#     echo "Please, provide the the training framework: torch/tf and dataset path"
#     exit 1
# fi


python scripts/run_pipeline.py torch -c ml3d/configs/pointpillars_kitti.yml \
--dataset_path /home/amlab/KITTI_point --pipeline ObjectDetection
