#!/bin/bash
#SBATCH --job-name="Oracles"
#SBATCH --output="./Projects/CDP/scripts/nav2d/oracles/out/o.out"
#SBATCH --error="./Projects/CDP/scripts/nav2d/oracles/out/o.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=9

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 ; do
    echo "Running script "
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method oracle_cdp -dir $1 -config $2 -file_name $3 -seed $seed -train True &
done
wait




