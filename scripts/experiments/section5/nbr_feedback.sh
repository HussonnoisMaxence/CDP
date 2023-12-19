nbr_feedbacks="256 128 16" 

for nbr_feedback in $nbr_feedbacks; do 
    cfg="extension/exp5/feedback/$nbr_feedback"
    for seed in 12345; do #23451 34512 45123 51234 67890 78906 89067 90678 6789
        python train_cdp.py cfg=$cfg cfg.seed=$seed 
    done
    wait
done