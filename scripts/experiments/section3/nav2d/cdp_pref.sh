for seed in 12345; do #23451 34512 45123 51234 67890 78906 89067 90678 6789
    python train_cdp.py cfg='extension/exp3/nav2d/cdp_pref' cfg.seed=$seed
done