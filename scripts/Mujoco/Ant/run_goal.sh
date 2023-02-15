#!/bin/bash
#SBATCH --job-name="AntG"
#SBATCH --output="./Projects/CDP/scripts/Mujoco/Ant/out/goal.out"
#SBATCH --error="./Projects/CDP/scripts/Mujoco/Ant/out/goal.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=10

declare dir='experiments/Mujoco/Ant/test4/'

for seed in 12345 23451 34512 45123 51234 67890 78906, 89067, 90678, 06789 ; do
    echo "Running script"
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method cdp  -dir $dir -config mujoco/ant/cdp.yaml -task goal -seed $seed &
done

wait 