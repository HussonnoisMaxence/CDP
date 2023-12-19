capacities="256 128 16" 

for capacity in $capacities; do 
    for seed in 12345; do #23451 34512 45123 51234 67890 78906 89067 90678 6789
        python train_cdp.py cfg="extension/exp5/capacity/cdp" cfg.seed=$seed cfg.rm_hidden_size=$capacity cfg.rm_last_size=$capacity
    done
    wait
done