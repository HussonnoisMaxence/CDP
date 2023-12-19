

humans="cdp equal_teacher mistake_teacher myopic_teacher noisy_teacher skip_teacher"  

for human in $humans; do 
    cfg="extension/exp5/human_model/$human"
    for seed in 12345; do #23451 34512 45123 51234 67890 78906 89067 90678 6789
        python train_cdp.py cfg=$cfg cfg.seed=$seed
    done
    wait
done

