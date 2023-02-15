#!/bin/bash
#SBATCH --job-name="HCRB"
#SBATCH --output="./Projects/CDP/scripts/Mujoco/HC/out/rback.out"
#SBATCH --error="./Projects/CDP/scripts/Mujoco/HC/out/rback.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=10

declare dir='experiments/Mujoco/HC/test4/'

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 06789 ; do
    echo "Running script"
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method cdp  -dir $1 -beta $2 -config mujoco/hc/cdpf.yaml -train True -task run_back -seed $seed &
done

wait 