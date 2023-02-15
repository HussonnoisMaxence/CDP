
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import itertools, os
from utils.utils import get_env
import torch.nn.functional as F
from utils.networks import eval_mode
import torch

def plot_rewardskill(reward_model, skill_number, logger, step, info=''):
    obs = np.array(list(itertools.permutations(np.arange(-1, 1, 2/255.0, dtype=float), 2)))

    nc=skill_number//4
    fig, axarr = plt.subplots(ncols=nc, nrows=4)
    for z in range (skill_number):               
        z_vector = np.zeros(skill_number)
        z_vector[z] = 1
        z_vector = np.repeat([z_vector],obs.shape[0], axis=0)
        obsz = np.concatenate([obs, z_vector], axis=-1)
        r = reward_model.r_hat_batch(obsz)


        obsp = obs
        x = obsp[:,0]
        y = obsp[:,1]
        obsp = [[o]for o in obsp]

        vmin = -1
        vmax = 1

        ax = axarr.flatten()[z]

        sc = ax.scatter(x, y, s=3, marker='o', c=r, vmin=vmin, vmax=vmax, cmap='bwr')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)



    for idx in range(skill_number):
            axarr.flatten()[idx].axis('off')
    cbar = fig.colorbar(sc,ax=axarr.ravel().tolist(), shrink=0.3, orientation="horizontal", pad=0.1)

    isExist = os.path.exists(logger.path+'/reward/')
    if not isExist:
            os.makedirs(logger.path+'/reward/')
    plt.savefig(logger.path+'/reward/'+info+str(step)+'.png')
    plt.clf()


def plot_reward(reward_model, logger, step, info=''):
    obs = np.array(list(itertools.permutations(np.arange(-1, 1, 2/255.0, dtype=float), 2)))

    z = reward_model.r_hat_batch(obs)


    obs = obs
    x = obs[:,0]
    y = obs[:,1]
    obs = [[o]for o in obs]

    vmin = -1
    vmax = 1
    fig, axarr = plt.subplots(ncols=1, nrows=1)

    ax = axarr

    sc = ax.scatter(x, y, s=3, marker='o', c=z, vmin=vmin, vmax=vmax, cmap='bwr')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title('rew', fontsize=14)




    cbar = fig.colorbar(sc, ax=axarr, shrink=0.3, orientation="horizontal", pad=0.1)

    isExist = os.path.exists(logger.path+'/reward/')
    if not isExist:
            os.makedirs(logger.path+'/reward/')
    plt.savefig(logger.path+'/reward/'+info+str(step)+'.png')
    plt.clf()


def plot_heatmapF(device, discriminator, logger, step=-1):
    obs = torch.tensor(np.array(list(itertools.permutations(np.arange(-255, 255, 2, dtype=float), 2))), dtype=torch.float)/255.0
    dataset = obs.to(device)

    obs = obs.detach().cpu().numpy()

    obs = obs*255.0
    x = obs[:,0]
    y = obs[:,1]
    obs = [[o]for o in obs]
    vmin = -5
    vmax = 0    

    nc=discriminator.codebook_size//2
    fig, axarr = plt.subplots(ncols=nc, nrows=2)
    vmax_qs = []
    for i in range(discriminator.codebook_size):
            z = torch.full((len(dataset),), i, dtype=torch.long).to(device)
            logprobs = discriminator.log_approx_posterior(dict(next_state=dataset, skill=z)).detach().cpu().numpy()
            vmin = min(logprobs)
            vmax = max(logprobs)
            idxs = np.where(logprobs==vmax)
            #print(idxs)
            
            ax = axarr.flatten()[i]
            
            sc = ax.scatter(x, y, s=3, marker='o', c=logprobs, cmap='bwr')
            ax.set_xlim(-255, 255)
            ax.set_ylim(-255, 255)
            ax.set_title('z'+str(i), fontsize=14)
            vmax_qs.append((x[idxs], y[idxs]))
 
    for idx in range(discriminator.codebook_size):
            axarr.flatten()[idx].axis('off')

    cbar = fig.colorbar(sc, ax=axarr.ravel().tolist(), shrink=0.3, orientation="horizontal", pad=0.1)

    isExist = os.path.exists(logger.path+'/heatmapF/')
    if not isExist:
            os.makedirs(logger.path+'/heatmapF/')
    plt.savefig(logger.path+'/heatmapF/VQrewards'+str(step)+'.png')
    plt.clf()
    


def heat_mapVQ(device, discriminator, logger, step=-1):
    obs = torch.tensor(np.array(list(itertools.permutations(np.arange(-1, 1, 2/255.0, dtype=float), 2))), dtype=torch.float)
    dataset = obs.to(device)

    obs = obs.detach().cpu().numpy()

    obs = obs
    x = obs[:,0]
    y = obs[:,1]
    obs = [[o]for o in obs]
    vmin = -5
    vmax = 0    

    nc=discriminator.codebook_size//2
    fig, axarr = plt.subplots(ncols=nc, nrows=2)
    vmax_qs = []
    for i in range(discriminator.codebook_size):
            z = torch.full((len(dataset),), i, dtype=torch.long).to(device)
            logprobs = discriminator.compute_logprob_under_latent(dict(next_state=dataset, skill=z)).detach().cpu().numpy()
            vmin = min(logprobs)
            vmax = max(logprobs)
            idxs = np.where(logprobs==vmax)
            #print(idxs)
            
            ax = axarr.flatten()[i]
            
            sc = ax.scatter(x, y, s=3, marker='o', c=logprobs, vmin=vmin, vmax=vmax, cmap='bwr')
            sc = ax.scatter(x[idxs], y[idxs], s=9, marker='*', c='y', vmin=vmin, vmax=vmax,)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title('z'+str(i), fontsize=14)
            vmax_qs.append((x[idxs], y[idxs]))
 
    for idx in range(discriminator.codebook_size):
            axarr.flatten()[idx].axis('off')

    cbar = fig.colorbar(sc, ax=axarr.ravel().tolist(), shrink=0.3, orientation="horizontal", pad=0.1)

    isExist = os.path.exists(logger.path+'/heatmap/')
    if not isExist:
            os.makedirs(logger.path+'/heatmap/')
    plt.savefig(logger.path+'/heatmap/VQrewards'+str(step)+'.png')
    plt.clf()
    
    #######################################################################################################################################################

    fig, axarr = plt.subplots(ncols=2, nrows=1, figsize=(20,10))
    ax = axarr[0]
    color_mapping = []
    for emb_idx in (range(discriminator.codebook_size)):
        s = discriminator.get_centroids(dict(skill=torch.tensor(emb_idx).to(device)))[0].detach().cpu().numpy()
        #print(s)
        m = ax.plot(s[0], s[1], marker='o', markersize=5, label='Code #{}'.format(emb_idx))
        color_mapping.append(m[0].get_color())
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    color_mapping_rgb = [list(int(h.lstrip('#')[i:i+2], 16)for i in (0, 2, 4)) for h in color_mapping]
    
    legend = ax.legend(loc='upper right', bbox_to_anchor=(2.4, 1), fontsize=12, ncol=1)
    ax = axarr[1]
    z_q_x = discriminator.vq.quantize(discriminator.encoder(dataset)).detach().cpu().numpy()
    plt.scatter(x, y, c=[color_mapping[z] for z in z_q_x], s=3, marker='o')
    plt.scatter(x, y, c=[color_mapping[z] for z in z_q_x], s=3, marker='o')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1, 1)
    plt.savefig(logger.path+'/heatmap/SkillAssignement'+str(step)+'.png')
    plt.clf()

    return vmax_qs

def plot_centroid(discriminator,device, logger, info, step=-1):
    for emb_idx in (range(discriminator.codebook_size)):
        s = discriminator.get_centroids(dict(skill=torch.tensor(emb_idx).to(device)))[0].detach().cpu().numpy()
        print(s, emb_idx)

        if len(s) == 1:
            plt.scatter(s[0], emb_idx)
        else:
            plt.scatter(s[0], s[1], s=1)
    plt.legend()
    isExist = os.path.exists(logger.path+'/centroids/')
    if not isExist:
            os.makedirs(logger.path+'/centroids/')
    plt.savefig(logger.path+'/centroids/'+info+str(step)+'.png')
    plt.clf()




def plot_state(datas, logger, info, step=-1):
    for label, data in enumerate(datas):
        obs = data.detach().cpu().numpy()



        obs = obs
        x = obs[:,0]
        y = obs[:,1]


        plt.scatter(x, y, s=1,label=label)
    plt.axis([-1, 1, -1, 1])

    
    plt.legend()
    isExist = os.path.exists(logger.path+'/coverage/')
    if not isExist:
            os.makedirs(logger.path+'/coverage/')
    plt.savefig(logger.path+'/coverage/'+info+str(step)+'.png')
    plt.clf()

import matplotlib.colors as colors
def plot_skill(agent, env, logger, skill_number, training_config, vmax_qs=None, timestep=-1,infop=''):
    step_max = training_config['step_max']
    fig, ax = plt.subplots(ncols=1, nrows=1)

    for i in range (skill_number):
        ## plot variable
        X,Y = [], []
        #Initiate the episode
        obs = env.reset()
        done = False
        episode_step = 0
        #sample skills
        z_vector = np.zeros(skill_number)
        z_vector[i] = 1
        while not(done):
            with eval_mode(agent):
                action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
            obs, reward, done, info = env.step(action)
            done = True if episode_step + 1 == step_max else False
            #done = True if episode_step == step_max else done
            episode_step +=1
            X.append(obs[0])
            Y.append(obs[1])

        ax.plot(X, Y, label=str(i))
        if vmax_qs!=None:
            ax.plot(vmax_qs[i][0], vmax_qs[i][1], marker="*", label=str(i))
    plt.axis([-1, 1, -1, 1])
    #loger.plot(label=i, color=colors_list[i])
    isExist = os.path.exists(logger.path+'/skill'+infop+'/')
    if not isExist:
            os.makedirs(logger.path+'/skill'+infop+'/')
    plt.savefig(logger.path+'/skill'+infop+'/'+str(timestep)+'.png')
    plt.clf()
    print('***Plot skill***')


def plot_skillHC(agent, env, logger, skill_number, training_cfg, timestep=-1,infop=''):
    step_max = training_cfg['step_max']
    fig, axarr = plt.subplots(ncols=3, nrows=2)

    for i in range (skill_number):
        ## plot variable
        X0,Y0 = [], []
        X1,Y1 = [], []
        X2,Y2 = [], []
        X3,Y3 = [], []
        X4,Y4 = [], []
        X5,Y5 = [], []
        #Initiate the episode
        if training_cfg['prior']:
            obs, prior = env.reset()
        else:
            obs = env.reset()
        done = False
        episode_step = 0
        #sample skills
        z_vector = np.zeros(skill_number)
        z_vector[i] = 1
        while not(done):
            with eval_mode(agent):
                action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
            obs, reward, done, info = env.step(action)
            done = True if episode_step == step_max -1 else done
            #done = True if episode_step == step_max else done
            episode_step +=1
            #plot X pos
            X0.append(info[0])
            Y0.append(10*i)

            #plot Z pos
            X1.append(obs[0])
            Y1.append(10*i)
            #plot XZ pos
            X2.append(info[0])
            Y2.append(obs[0])
            #plot X s
            X3.append(obs[8])
            Y3.append(10*i)
            #plot Z s
            X4.append(obs[9])
            Y4.append(10*i)
            #plot XZ s
            X5.append(obs[8])
            Y5.append(obs[9])
        axarr[0][0].plot(X0, Y0, label=str(i))
        axarr[0][1].plot(X1, Y1, label=str(i))
        axarr[0][2].plot(X2, Y2, label=str(i))
        axarr[1][0].plot(X3, Y3, label=str(i))
        axarr[1][1].plot(X4, Y4, label=str(i))
        axarr[1][2].plot(X5, Y5, label=str(i))
    #loger.plot(label=i, color=colors_list[i])
    isExist = os.path.exists(logger.path+'/skill'+infop+'/')
    if not isExist:
            os.makedirs(logger.path+'/skill'+infop+'/')
    plt.savefig(logger.path+'/skill'+infop+'/'+str(timestep)+'.png')
    plt.clf()
    print('***Plot skill***')
from gym.wrappers import RecordVideo
def run_test_videos(agent, env_name, logger, skill_number, training_cfg, info=''):
    step_max = training_cfg['step_max']
    for i in range (skill_number):
        #Initiate the episode
        env = RecordVideo(get_env(env_name), logger._log_dir + "/videos/"+info+"/skill-"+str(i))
        if training_cfg['prior']:
            obs, prior = env.reset()
        else:
            obs = env.reset()
        done = False
        episode_step = 0
        #sample skills
        z_vector = np.zeros(skill_number)
        z_vector[i] = 1
        while not(done):
            with eval_mode(agent):
                action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
            obs, reward, done, _ = env.step(action)
            done = True if episode_step == step_max-1 else done
            episode_step +=1
        print('step', episode_step)
    print('***End test***')

import math
def target_reward(pos, goal=[200/255.0,200/255.0]):
    dist = math.sqrt(pow((goal[0] - pos[0]), 2) + pow((goal[1]  - pos[1]), 2))
    norm = 1 #math.sqrt(pow((2), 2) + pow((2), 2))
    done = bool(dist <= 5/255.0)
    if done:
        return 1
    else:
        return -dist/norm


def plot_klsm(ps, device, logger):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    obs = np.array(list(itertools.permutations(np.arange(-255, 255, 2, dtype=float), 2)))/255.0
    
    values = [target_reward(o) for o in obs]
    mean = np.mean(values)
    ma = np.max(values)
    mi = np.min(values)
    t = np.std(values)
    print(mean, ma, mi, t)
    values = [(v-mi)/(ma-mi) for v in values]
    mean = np.mean(values)
    ma = np.max(values)
    mi = np.min(values)
    t = np.std(values)
    print(mean, ma, mi, t)
    idxs = np.where(values > np.array([0.55]))[0]
    idxs_t = torch.from_numpy(idxs).long()
    obs  = torch.tensor(obs, dtype=torch.float)
    target_ps = torch.index_select(obs, 0, idxs_t)
    v = torch.tensor(values, dtype=torch.float)
    target_mean = torch.mean(torch.index_select(v, 0, idxs_t))
    print(target_mean)
    plot_state([target_ps], logger, info='target', step=-1)
    print(a)
    indices = list(range(target_ps.size(0)))
    batch = ps.size(0)
    mean = 0
    for i in range(1000):
        batch_indices = np.random.choice(indices, size=batch)
        input = ps

        target = target_ps[batch_indices].to(device)
        mean += kl_loss(input, target).item()
    return mean/1000.0

def get_targetset(device):
    return torch.tensor(np.array(list(itertools.permutations(np.arange(100, 255, 2, dtype=float), 2)))/255.0).to(device), np.array([75, 255])/255.0


def get_ws_metric(target, obs):
    indices = list(range(target.size(0)))
    batch_indices = np.random.choice(indices, size=obs.size(0))
    return calculate_2_wasserstein_dist(obs, target[batch_indices])


def get_kl_metric(input, target):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    return kl_loss(input, target)

def get_metric(obs, target, goal, t):
    ## get state that match the target state space
    idxs = obs[(obs[:,0] > goal[0]) & (obs[:,1] > goal[0]) & (obs[:,0] < goal[1]) & (obs[:,1] < goal[1])]
    allignment_metric =  len(idxs)/t
    ## get state that match the target state space
    indices = list(range(target.size(0)))
    batch_indices = np.random.choice(indices, size=obs.shape[0])

    hs = compute_state_entropy(target[batch_indices], obs, k=15)

    coverage_metric = torch.mean(hs)
    #coverage_metric_std = torch.var(hs)
    l = []
    for i,o in enumerate(obs):
        h_s = compute_state_entropy(o.unsqueeze(0),torch.cat([obs[0:i,:], obs[i+1:,:]]), k=5)
        l.append(h_s.detach().cpu().numpy())

    
    coverage_metric_std = np.mean(l)
    return allignment_metric, coverage_metric, coverage_metric_std

def get_allignment_metric(obs, goal):
    
    p =  0
    return p #*p

def get_coverage_metric(obs, target):
    return torch.mean(compute_state_entropy(target, obs, k=2))

def get_coverage_metric2(obs, goal):
    idxs = obs[(obs[:,0] > goal[0]) & (obs[:,1] > goal[0]) & (obs[:,0] <goal[1]) & (obs[:,1] < goal[1])]
    p =   len(idxs)/obs.shape[0] + 0.000001

    l = []
    for i,o in enumerate(obs):
        h_s = compute_state_entropy(o.unsqueeze(0),torch.cat([obs[0:i,:], obs[i+1:,:]]), k=5)
        l.append(h_s.detach().cpu().numpy())

    
    print(np.mean(l),p)

    return 0
def measure(logger):
    obs = np.array(list(itertools.permutations(np.arange(-255, 255, 2, dtype=float), 2)))/255.0
    tobs = torch.tensor(np.array(list(itertools.permutations(np.arange(100, 255, 2, dtype=float), 2)))/255.0)
    tobs2 = torch.tensor(np.array(list(itertools.permutations(np.arange(100, 150, 2, dtype=float), 2)))/255.0)
    tobs6 = torch.tensor(np.array(list(itertools.permutations(np.arange(50, 120, 2, dtype=float), 2)))/255.0)

    get_coverage_metric2(tobs, goal=np.array([100, 255])/255.0)
    get_coverage_metric(tobs, tobs, goal=np.array([100, 255])/255.0)

    get_coverage_metric2(tobs2, goal=np.array([100, 255])/255.0)
    get_coverage_metric(tobs, tobs2, goal=np.array([100, 255])/255.0)

    get_coverage_metric2(tobs6, goal=np.array([100, 255])/255.0)
    get_coverage_metric(tobs, tobs6, goal=np.array([100, 255])/255.0)

    print(a)
    tobs3 = torch.tensor(np.array(list(itertools.permutations(np.arange(200, 250, 2, dtype=float), 2)))/255.0)
    tobs4 = torch.tensor(np.array(list(itertools.permutations(np.arange(100, 255, 3, dtype=float), 2)))/255.0)
    tobs5 = torch.tensor(np.array(list(itertools.permutations(np.arange(100, 120, 2, dtype=float), 2)))/255.0)
    tobs6 = torch.tensor(np.array(list(itertools.permutations(np.arange(50, 120, 2, dtype=float), 2)))/255.0)







    print(tobs6.shape, tobs2.shape)
    plot_state([tobs2, tobs3], logger, info='target2', step=-1)
    plot_state([tobs], logger, info='target3', step=-1)
    plot_state([tobs4], logger, info='target23', step=-1)
    plot_state([tobs5], logger, info='target237', step=-1)
    sentropy = compute_state_entropy(tobs, tobs, k=5)
    sentropy1 = compute_state_entropy(tobs,tobs2, k=5)
    sentropy2 = compute_state_entropy(tobs, tobs3, k=5)
    sentropy3 = compute_state_entropy(tobs2, tobs3, k=5)
    sentropy4 = compute_state_entropy(tobs, tobs4,  k=5)
    sentropy5 = compute_state_entropy(tobs, tobs5,  k=5)
    sentropy6 = compute_state_entropy(tobs, tobs6,  k=5)
    print(torch.mean(sentropy), torch.mean(sentropy1),torch.mean(sentropy2),torch.mean(sentropy3), 
    torch.mean(sentropy4),torch.mean(sentropy5),torch.mean(sentropy6)) #,torch.mean(sentropy4))
    
    t=np.array([100, 255])/255.0
    tobs5=tobs5.numpy()
    idxs = tobs5[(tobs5[:,0] > t[0]) & (tobs5[:,1] > t[0]) & (tobs5[:,0] < t[1]) & (tobs5[:,1] < t[1])]#np.where(tobs5[:,0] > t[0] and tobs5[:,1] > t[0] and tobs5[:,0] < t[1] and tobs5[:,1] < t[1])
    p5 =  1 - len(idxs)/tobs5.shape[0] + 0.000001
    idxs = tobs6[(tobs6[:,0] > t[0]) & (tobs6[:,1] > t[0]) & (tobs6[:,0] < t[1]) & (tobs6[:,1] < t[1])]
    p6 =  1 - len(idxs)/tobs6.shape[0] + 0.000001
    print(torch.mean(sentropy5)*(p5),torch.mean(sentropy6)*(p6)) #,torch.mean(sentropy4))
    print(a)
    plot_state([torch.tensor(tobs)], logger, info='target2', step=-1)
    ##target state
    values = [target_reward(o) for o in obs]
    ma = np.max(values)
    mi = np.min(values)
    values = [(v-mi)/(ma-mi) for v in values]
    idxs = np.where(values > np.array([0.5]))[0]
    idxs_t = torch.from_numpy(idxs).long()
    target_state = torch.index_select(torch.tensor(obs), 0, idxs_t)
    ##sample state
    idxs = np.where(values > np.array([0.7]))[0]
    idxs_t = torch.from_numpy(idxs).long()
    p_state = torch.index_select(torch.tensor(obs), 0, idxs_t)
    idxs = np.where(values > np.array([0.9]))[0]
    idxs_t = torch.from_numpy(idxs).long()
    p2_state = torch.index_select(torch.tensor(obs), 0, idxs_t)
    obs = torch.tensor(obs)
    indices = list(range(obs.size(0)))       
    batch_indices = np.random.choice(indices, size=256)
    samples_state = torch.tensor(obs[batch_indices])
    plot_state([target_state], logger, info='target', step=-1)
    plot_state([p_state], logger, info='sample', step=-1)

    sentropy = compute_state_entropy(target_state, target_state, k=5)
    sentropy1 = compute_state_entropy(target_state, p_state, k=5)
    sentropy2 = compute_state_entropy(target_state, p2_state, k=5)
    sentropy3 = compute_state_entropy(samples_state, target_state, k=5)
    sentropy4 = compute_state_entropy(target_state, samples_state,  k=5)
    print(torch.mean(sentropy), torch.mean(sentropy1),torch.mean(sentropy2),torch.mean(sentropy3),torch.mean(sentropy4))
    
def compute_state_entropy(obs, full_obs, k):
    batch_size = 500
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(
                obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2
            )
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1)

def compute_density(obs, goal):
    idxs = obs[(obs[:,0] > goal[0]) & (obs[:,1] > goal[0]) & (obs[:,0] <goal[1]) & (obs[:,1] < goal[1])]
    return 1-len(idxs)/obs.shape[0]

def plot_density(logger, x, y):




    plt.plot(x, y)
    
    plt.legend()
    isExist = os.path.exists(logger.path+'/coverage/')
    if not isExist:
            os.makedirs(logger.path+'/coverage/')
    plt.savefig(logger.path+'/coverage/'+'density.png')
    plt.clf()




def plot_skill_prior(agent, env, logger, skill_number, training_config, timestep=-1,infop=''):
    step_max = training_config['step_max']
    fig, ax = plt.subplots(ncols=1, nrows=1)
    for i in range (skill_number):
        ## plot variable
        X,Y = [], []
        #Initiate the episode
        if training_config['prior']:
            obs, prior = env.reset()
        else:
            obs = env.reset()
        done = False
        episode_step = 0
        #sample skills
        z_vector = np.zeros(skill_number)
        z_vector[i] = 1
        while not(done):
            with eval_mode(agent):
                action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
            obs, reward, done, info = env.step(action)
            done = True if episode_step == step_max -1 else done
            #done = True if episode_step == step_max else done
            episode_step +=1
            if len(info)==1:
                X.append(info[0])
                Y.append(i)
            else:
                X.append(info[0])
                Y.append(info[1])
        ax.plot(X, Y, label=str(i))
    #loger.plot(label=i, color=colors_list[i])
    isExist = os.path.exists(logger.path+'/skill'+infop+'/')
    if not isExist:
            os.makedirs(logger.path+'/skill'+infop+'/')
    plt.savefig(logger.path+'/skill'+infop+'/'+str(timestep)+'.png')
    plt.clf()
    print('***Plot skill***')

def save_data(logger, data, name):
    
    isExist = os.path.exists(logger._log_dir+'/data/')
    if not isExist:
            os.makedirs(logger._log_dir+'/data/')
    np.save(logger._log_dir+'/data/'+name, data)

