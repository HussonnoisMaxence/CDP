#!/bin/bash
#SBATCH --job-name="HCRBbl"
#SBATCH --output="./Projects/CDP/scripts/Mujoco/HC/out/rbackbl.out"
#SBATCH --error="./Projects/CDP/scripts/Mujoco/HC/out/rbackbl.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=10

declare dir='experiments/Mujoco/HC/test4/'

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 06789 ; do
    echo "Running script"
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method edl  -dir $1  -config mujoco/hc/cdp_edl.yaml -train True -task run_back -seed $seed &
done

wait 