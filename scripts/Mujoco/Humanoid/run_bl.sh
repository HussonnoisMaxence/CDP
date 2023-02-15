#!/bin/bash
#SBATCH --job-name="HumBL"
#SBATCH --output="./Projects/CDP/scripts/Mujoco/Humanoid/out/BL.out"
#SBATCH --error="./Projects/CDP/scripts/Mujoco/Humanoid/out/BL.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=10

declare dir='experiments/Mujoco/Ant/test4/'

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 06789 ; do
    echo "Running script"
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method cdp -dir $1 -beta $2 -config mujoco/hum/cdp.yaml -task forward -train True -seed $seed &
done

wait 