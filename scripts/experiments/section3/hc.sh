python main.py -method cdp  -dir experiments/section4/hc/ -file_name state -beta 0.9 -config mujoco/hc/cdp.yaml -beta 0.9 -task run_back -seed 90678 -train True
python main.py -method cdp  -dir experiments/section4/hc/ -file_name prior_speed -beta 0.9 -config mujoco/hc/cdp_prior.yaml -beta 0.9 -task run_back -seed 90678 -train True
python main.py -method cdp  -dir experiments/section4/hc/ -file_name pref -beta 0.9 -config mujoco/hc/cdpf.yaml -beta 0.9 -task run_back -seed 6789 -train True