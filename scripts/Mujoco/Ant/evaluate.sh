#!/bin/bash
#SBATCH --job-name="AntEv"
#SBATCH --output="./Projects/CDP/scripts/Mujoco/Ant/out/ev.out"
#SBATCH --error="./Projects/CDP/scripts/Mujoco/Ant/out/ev.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=10

declare dir='experiments/Mujoco/HC/test4/'

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 06789 ; do
    echo "Running script"
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method cdp  -dir $1 -beta $2 -config mujoco/ant/cdp_feature.yaml -task forward -seed $seed -eval True &
done

wait 