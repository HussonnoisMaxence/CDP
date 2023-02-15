# CDP

## Installation
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e ./envs/gym-nav2d/

## Run section 1 experiments
sh ./scripts/experiments/section1/cdp.sh
sh ./scripts/experiments/section1/edl.sh

## Run section 2 experiments
sh ./scripts/experiments/section2/cdp.sh
sh ./scripts/experiments/section2/smm.sh
sh ./scripts/experiments/section2/smm_prior.sh
sh ./scripts/experiments/section2/plots.sh

## Run section 3 experiments
sh ./scripts/experiments/section3/hc.sh
sh ./scripts/experiments/section3/nav2d.sh

## Run section 4 experiments
sh ./scripts/experiments/section4/nav2d.sh
sh ./scripts/experiments/section4/hc.sh

