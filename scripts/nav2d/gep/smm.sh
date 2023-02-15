#!/bin/bash
#SBATCH --job-name="GEPs"
#SBATCH --output="./Projects/CDP/scripts/nav2d/gep/out/geps.out"
#SBATCH --error="./Projects/CDP/scripts/nav2d/gep/out/geps.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=9

declare dir='experiments/nav2d/gep/test2/'

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 ; do
    echo "Running script "
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method cdp -dir $1 -file_name smm -config nav2d/gep/smm.yaml -seed $seed -train True &
done
wait