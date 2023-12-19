# Human-Informed Skill Discovery: Controlled Diversity with Preference in Reinforcement Learning
Authors: Maxence Hussonnois, Thommen George Karimpanal, Mayank Shekhar JHA and Santu Rana
Link to paper: under review

## Abstract
Autonomously learning diverse behaviors without an extrinsic reward signal has been a problem of interest in reinforcement learning. 
However, the nature of learning in such mechanisms is unconstrained, often resulting in the accumulation of several unuseful, unsafe or misaligned skills. In order to avoid such issues and to ensure the discovery of safe and human-aligned skills, it is necessary to incorporate humans into the unsupervised training process, which remains a largely unexplored topic.
In this work, we propose Controlled Diversity with Preference (CDP), a novel, collaborative human-guided mechanism for an agent to learn a set of skills that is diverse as well as desirable. The key principle is to restrict the discovery of skills to regions that are deemed to be desirable as per a preference model trained using human preference labels on trajectory pairs. We evaluate our approach on 2D navigation and Mujoco environments and demonstrate the ability to discover diverse, yet desirable skills. We also provide principled guidelines for selecting suitable hyperparameter values along with comprehensive sensitivity analyses of the various factors influencing the performance of our approach.

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
### SECTION 1 Results in 2D Navigation

```
sh scripts/experiments/section1/cdp.sh
```
```
sh scripts/experiments/section1/edl.sh
```
### SECTION 2 Exploration of the Preferred Region
```
sh scripts/experiments/section2/cdp.sh
```
```
sh scripts/experiments/section2/smm.sh
```
```
sh scripts/experiments/section2/smmp.sh
```
```
sh scripts/experiments/section2/smmp_vqvae.sh
```

### SECTION 3 Preferred Latent Representation
```
sh scripts/experiments/section3/nav2d/cdp_pref.sh
```
```
sh scripts/experiments/section3/hc/plr.sh
```
```
sh scripts/experiments/section3/hc/prior.sh
```
```
sh scripts/experiments/section3/hc/state.sh
```


### SECTION 4 Analysis of the preference reward threshold Beta

```
sh scripts/experiments/section4/section42/nav2d.sh
```
```
sh scripts/experiments/section4/section43/hc.sh
```
```
sh scripts/experiments/section4/section43/nav2d.sh
```



### SECTION 5 Sensitivity Analysis Reward Model

```
sh scripts/experiments/section5/capacity.sh
```
```
sh scripts/experiments/section5/nbr_feedback.sh
```
```
sh scripts/experiments/section5/sampling.sh
```
```
sh scripts/experiments/section5/teachers.sh
```