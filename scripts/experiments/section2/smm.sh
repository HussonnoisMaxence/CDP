for seed in 12345; do #23451 34512 45123 51234 67890 78906 89067 90678 6789
    python train_cdp.py cfg='extension/exp2/smm' cfg.seed=$seed
done