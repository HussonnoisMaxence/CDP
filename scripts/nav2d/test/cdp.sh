#!/bin/bash
#SBATCH --job-name="Test_cdp"
#SBATCH --output="./Projects/CDP/scripts/nav2d/test/out/cdp.out"
#SBATCH --error="./Projects/CDP/scripts/nav2d/test/out/cdp.err"
#SBATCH --partition=gpu



python ./Projects/CDP/main.py -method cdp -dir experiments/nav2d/TB/ -file_name beta -config nav2d/tests/results.yaml
