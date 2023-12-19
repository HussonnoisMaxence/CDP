betas="0.6 0.75 0.9"  

for beta in $betas; do 
    for seed in 12345; do #23451 34512 45123 51234 67890 78906 89067 90678 6789
        python train_cdp.py cfg="extension/exp4/betas/fix" cfg.seed=$seed cfg.beta_pref=$beta
    done
    wait
done
wait

for seed in 12345; do #23451 34512 45123 51234 67890 78906 89067 90678 6789
    python train_cdp.py cfg="extension/exp4/betas/adj" cfg.seed=$seed
done
wait