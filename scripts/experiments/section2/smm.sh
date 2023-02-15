for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    python main.py -method cdp -dir experiments/section2/ -config nav2d/gep/smm.yaml -file_name smm -seed $seed -train True
done