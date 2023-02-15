#!/bin/bash
#SBATCH --job-name="GEPsp"
#SBATCH --output="./Projects/CDP/scripts/nav2d/gep/out/gepsp.out"
#SBATCH --error="./Projects/CDP/scripts/nav2d/gep/out/gepsp.err"
#SBATCH --partition=gpu
#SBATCH --ntasks=9



for seed in 12345 23451 34512 45123 51234 67890 78906 89067 ; do
    echo "Running script "
    srun --exclusive --ntasks=1 python ./Projects/CDP/main.py -method cdp -dir $1 -file_name smm_prior -config nav2d/gep/smm_prior.yaml -seed $seed -train True &
done
wait