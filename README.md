# Controlled Diversity with Preference : Towards Learning a Diverse Set of Desired Skills
Authors: Maxence Hussonnois, Thommen George Karimpanal, Santu Rana
Link to paper: todo

## Abstract
Autonomously learning diverse behaviors without an extrinsic
reward signal has been a problem of interest in reinforcement
learning. However, the nature of learning in such mechanisms is
unconstrained, often resulting in the accumulation of several un-
usable, unsafe or misaligned skills. In order to avoid such issues
and ensure the discovery of safe and human-aligned skills, it is
necessary to incorporate humans into the unsupervised training
process, which remains a largely unexplored research area. In this
work, we propose Controlled diversity with Preference (CDP), a
novel, collaborative human-guided mechanism for an agent to learn
a set of skills that is diverse as well as desirable. The key principle
is to restrict the discovery of skills to those regions that are deemed
to be desirable as per a preference model trained using human pref-
erence labels on trajectory pairs. We evaluate our approach on 2D
navigation and Mujoco environments and demonstrate the ability
to discover diverse, yet desirable skills.

## Installation
Create a virtual environment and install the packages listed in requirements.text and install gym-nav2d environment
```
python3 -m venv env
```
```
source env/bin/activate
```
```
pip install -r requirements.txt
```
```
pip install -e ./envs/gym-nav2d/
```
## Run experiments from the paper
### Run section 1 experiments

```
sh ./scripts/experiments/section1/cdp.sh
```
```
sh ./scripts/experiments/section1/edl.sh
```
### Run section 2 experiments

```
sh ./scripts/experiments/section2/cdp.sh
```
```
sh ./scripts/experiments/section2/smm.sh
```
```
sh ./scripts/experiments/section2/smm_prior.sh
```
```
sh ./scripts/experiments/section2/plots.sh
```

### Run section 3 experiments
```
sh ./scripts/experiments/section3/hc.sh
```
```
sh ./scripts/experiments/section3/nav2d.sh
```
### Run section 4 experiments
```
sh ./scripts/experiments/section4/nav2d.sh
```
```
sh ./scripts/experiments/section4/hc.sh
```
