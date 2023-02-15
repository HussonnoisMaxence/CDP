#!/bin/bash
#SBATCH --job-name="GEP-bf"
#SBATCH --output="./Projects/CDP/scripts/nav2d/gep/out/betaf.out"
#SBATCH --error="./Projects/CDP/scripts/nav2d/gep/out/betaf.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=9

declare dir='experiments/nav2d/beta/test1/'

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 ; do
    echo "Running script "
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method cdp -dir $1 -config nav2d/gep/beta/cdp_feat.yaml -seed $seed -beta $2 -train True &
done   
wait