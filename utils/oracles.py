

import itertools
import torch
import numpy as np

import math

from utils.networks import eval_mode
def target_reward_norm2(pos, goal=[200.0,200]):
    dist = math.sqrt(pow((goal[0] - pos[0]), 2) + pow((goal[1]  - pos[1]), 2))
    #norm = 360.62 #math.sqrt(pow((2), 2) + pow((2), 2))
    done = bool(dist <= 5)
    if done:
        return 1
    else:
        return -dist/644.0

def target_reward_norm(pos, goal=[200.0,200]):
    dist = math.sqrt(pow((goal[0] - pos[0]), 2) + pow((goal[1]  - pos[1]), 2))
    done = bool(dist <= 5)

    return 1 - dist/644.0

def target_reward(pos, goal=[200.0,200]):
    dist = math.sqrt(pow((goal[0] - pos[0]), 2) + pow((goal[1]  - pos[1]), 2))
    done = bool(dist <= 2.5)
    if done:
        return 1
    else:
        return -dist

def focus(states, values, rts, with_values=False):
        ma = np.max(values)
        mi = np.min(values)
        
        norm_values = np.array([(v-mi)/(ma-mi) for v in values])
        idxs = np.where(norm_values > rts)[0]
        idxs_t = torch.from_numpy(idxs).long()
        if with_values:
            norm_values = torch.from_numpy(norm_values)
            return  torch.index_select(states, 0, idxs_t)
        else:
            return  torch.index_select(states, 0, idxs_t)
import torch.nn.functional as F
def focus_strict(states, values, rts):
        ma = np.max(values)
        mi = np.min(values)
        mean=np.mean(values)
        norm_values = np.array(values)
        idxs = np.where(norm_values > rts)[0]
        idxs_t = torch.from_numpy(idxs).long()
        return  torch.index_select(states, 0, idxs_t)



def compute_target_rewards(obs, exp):
    return np.array([target_reward_norm(o*255.0) for o in obs] )

def get_preferred_region(obs, beta, values, relative=True):
    if not(relative):
        return focus_strict(obs, values, beta)
    else:
        return focus(obs, values, beta)

def get_distribution(input):
    input =  np.array([target_reward_norm(o*255.0) for o in input] ) #compute_target_rewards(input, exp=False)
    ma =  1
    mi = 0 #-1
    norm_values = torch.tensor([(v-mi)/(ma-mi) for v in input])#torch.tensor(input).float() #([(v-mi)/(ma-mi) for v in input]))
    return norm_values/torch.sum(norm_values)

def sample_target_region(batch, target):
    indices = list(range(target.size(0)))
    batch_indices = np.random.choice(indices, size=batch, replace=True)
    return target[batch_indices]

def get_kl_div(p_input, p_target):
    ## Compute Kl_divergence
    print(p_input.shape, p_target.shape)
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    return kl_loss(torch.log( p_input),p_target) #kl_loss(torch.log( F.softmax(p_input)), F.softmax(p_target))

def get_target_state(exp_name, rts):
    if exp_name=='TR_2d_nav':
        goal=np.array([200.0,200])
        full_state_coverage = torch.tensor(np.array(list(itertools.permutations(np.arange(-255, 255, 1, dtype=float), 2))), dtype=torch.float)
        values =[target_reward_norm(o, goal) for o in full_state_coverage]
        target_states = focus_strict(full_state_coverage/255.0, values, rts)

        #limits = np.array([torch.min(target_states[:,0]).item(),torch.min(target_states[:,1]).item(),torch.max(target_states[:,1]).item(),torch.max(target_states[:,1]).item()])
        limits= [0.3,1]
        return target_states, limits
#################################3
def get_preferred_region2(obs, beta, target, values=None):
    if target:
        goals= np.array([200.0,200])
        print('O')
        values = [target_reward(o*255.0, goals) for o in obs] 
        return focus2(obs, values, beta)
    else:
        return focus2(obs, values, beta)

def get_distribution2(input):
    goal=np.array([200.0,200])
    values = torch.tensor([target_reward(o*255.0, goal) for o in input])
    print(torch.mean(values), torch.std(values))
    return torch.mean(values), torch.std(values) #F.softmax(torch.tensor(values), dim=-1) #values/values.sum()#F.softmax(torch.tensor(values), dim=-1)


def focus2(states, values, rts):
        ma = np.max(values)
        mi = np.min(values)
        ma = ma
        mi = mi
        
        norm_values = np.array([(v-mi)/(ma-mi) for v in values])
        ma = np.max(norm_values)
        mi = np.min(norm_values)
        idxs = np.where(norm_values > rts)[0]
        idxs_t = torch.from_numpy(idxs).long()
        return  torch.index_select(states.detach().cpu(), 0, idxs_t)


def get_target_state2(exp_name, rts):
    if exp_name=='TR_2d_nav':
        goals= np.array([200.0,200])
        full_state_coverage = torch.tensor(np.array(list(itertools.permutations(np.arange(-255, 255, 1, dtype=float), 2))), dtype=torch.float)
        values = [target_reward(o, goals) for o in full_state_coverage] 
        states_p = focus2(full_state_coverage/255.0, values,  rts)  

        return states_p

def get_oracle(exp_name):
    if exp_name=='TR_2d_nav':
        goals=np.array([[200.0,200]])
        full_state_coverage = torch.tensor(np.array(list(itertools.permutations(np.arange(-255, 255, 1, dtype=float), 2))), dtype=torch.float)
        values =[ [target_reward(o, goal) for o in full_state_coverage] for goal in goals]
        return full_state_coverage/255.0, values, goals/255.0
    if exp_name=='edl':
            goals=np.array([[200.0,200]])
            full_state_coverage = torch.tensor(np.array(list(itertools.permutations(np.arange(-255, 255, 1, dtype=float), 2))), dtype=torch.float)
            values =[ [target_reward(o, goal) for o in full_state_coverage] for goal in goals]
            return full_state_coverage/255.0, values, goals/255.0
    if exp_name=='2OG_2d_nav':
        goals=np.array([[200.0,200], [-200.0,-200]])
        full_state_coverage = torch.tensor(np.array(list(itertools.permutations(np.arange(-255, 255, 1, dtype=float), 2))), dtype=torch.float)
        values =[ [target_reward(o, goal) for o in full_state_coverage] for goal in goals]
        return full_state_coverage/255.0, values, goals/255.0



from torch import distributions as pyd



import numpy as np
import matplotlib.pyplot as plt
import itertools, os
import torch

def plot_discriminator(device, discriminator, logger, step=-1):
    obs = torch.tensor(np.array(list(itertools.permutations(np.arange(-1, 1, 1/255, dtype=float), 2))), dtype=torch.float)
    dataset = obs.to(device)
    obs = obs.detach().cpu().numpy()
    x = obs[:,0]
    y = obs[:,1]
    obs = [[o]for o in obs]
    vmin = -5
    vmax = 0    
    plt.figure(figsize=(10.4, 8.8), dpi=500 )

    color_mapping = []    
    for emb_idx in (range(discriminator.codebook_size)):
        s = discriminator.get_centroids(dict(skill=torch.tensor(emb_idx).to(device)))[0].detach().cpu().numpy()

        m = plt.plot(s[0], s[1], marker='o', markersize=5, label='Code #{}'.format(emb_idx))
        color_mapping.append(m[0].get_color())
    plt.clf()
    color_mapping_rgb = [list(int(h.lstrip('#')[i:i+2], 16)for i in (0, 2, 4)) for h in color_mapping]


    z_q_x = discriminator.vq.quantize(discriminator.encoder(dataset)).detach().cpu().numpy()
    c = []
    plt.scatter(x, y, c=[color_mapping[z] for z in z_q_x], s=1, marker='o')
    for emb_idx in (range(discriminator.codebook_size)):
        s = discriminator.get_centroids(dict(skill=torch.tensor(emb_idx).to(device)))[0].detach().cpu().numpy()
        #print(s)
        m = plt.scatter(s[0], s[1],  edgecolor='black', linewidth=0.7,linestyle=':',  label='Skill {}'.format(emb_idx+1))
        c.append(s)
    plt.axis([-1, 1, -1, 1])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   mode="expand", borderaxespad=0., ncol=6, fontsize='xx-large') 
    isExist = os.path.exists(logger._log_dir+'/heatmap/')
    if not isExist:
            os.makedirs(logger._log_dir+'/heatmap/')
    plt.savefig(logger._log_dir+'/heatmap/SkillAssignement'+str(step)+'.pdf')
    plt.clf()
    plt.close()
    return c, color_mapping

def plot_state(datas, logger, info, goals, step=-1):
    
    plt.figure(figsize=(10.4, 8.8), dpi=500 )
    for label, data in enumerate(datas):
        obs = data.detach().cpu().numpy()
        obs = obs
        x = obs[:,0]
        y = obs[:,1]
        plt.scatter(x, y, s=1,label='preferred states')
    
    plt.scatter(0, 0, s=40, marker='+', color='r', label='initial state')
    if goals != []:
        for goal in goals:
            plt.scatter(goal[0], goal[1], s=40, marker='*', color='y', label='goal-center')
    plt.axis([-1, 1, -1, 1])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      mode="expand", borderaxespad=0., ncol=3, fontsize='xx-large') 
    isExist = os.path.exists(logger._log_dir+'/coverage/')
    if not isExist:
            os.makedirs(logger._log_dir+'/coverage/')
    plt.savefig(logger._log_dir+'/coverage/'+info+str(step)+'.pdf')
    plt.clf()
    plt.close()

def plot_skill(agent, env, logger, skill_number, training_config, vmax_qs=None, cm=None, timestep=-1,infop=''):
    step_max = training_config['step_max']
    plt.figure(figsize=(10.4, 8.8), dpi=500 )

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
            episode_step +=1
            X.append(obs[0])
            Y.append(obs[1])
        if vmax_qs==None:
            plt.plot(X, Y, label='Skill '+str(i), zorder=-1)
        else:
            plt.plot(X, Y, color=cm[i], label='Skill '+str(1+i), zorder=-1)

            plt.scatter(vmax_qs[i][0], vmax_qs[i][1],s=40, color=cm[i], marker="*", edgecolor='black', linewidth=0.7,zorder=1)

    plt.axis([-1, 1, -1, 1])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                mode="expand", borderaxespad=0., ncol=6, fontsize='xx-large') #labelspacing=1,  fontsize=20)

    isExist = os.path.exists(logger._log_dir+'/skill'+infop+'/')
    if not isExist:
            os.makedirs(logger._log_dir+'/skill'+infop+'/')
    plt.savefig(logger._log_dir+'/skill'+infop+'/'+str(timestep)+'.pdf')
    plt.clf()
    plt.close()
    print('***Plot skill***')





from PIL import Image, ImageDraw
def plot_skill_pos_ant(agent, env, logger, skill_number, training_config, discriminator, device, timestep=-1,infop=''):
    step_max = 1000 #training_config['step_max']
    plt.figure(figsize=(6.4, 4.8), dpi=500 )
    img = Image.new("RGBA", (6400, 6400), (255, 255, 255, 1))
    draw = ImageDraw.Draw(img)
    color = ['green', 'red', 'blue', 'orange', 'yellow','cyan', 'purple', 'brown', 'gray', 'olive']
    for i in range (skill_number):
        ## plot variable
        X,Y = [], []
        #Initiate the episode
        obs, prior = env.reset()
        done = False
        episode_step = 0
        #sample skills
        z_vector = np.zeros(skill_number)
        z_vector[i] = 1
        while not(done):
            with eval_mode(agent):
                action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
            obs, reward, done, prior = env.step(action)
            done = True if episode_step + 1 == step_max else done
            #done = True if episode_step == step_max else done
            
            X.append(env.sim.data.qpos[0])
            Y.append(env.sim.data.qpos[1])
            if not(episode_step%70):

                o = env.render(mode='rgb_array')

                h, w = o.shape[:2]
                n = np.dstack((o, np.zeros((h,w),dtype=np.uint8)+255))
                # Make mask of black pixels - mask is True where image is black
                mBlack = (n[:, :, 0:3] == [0,0,0]).all(2)
                #print(mBlack)
                n[mBlack] = (0,0,0,0)
                im = Image.fromarray(n)

                new_width  = 300
                width, height = im.size
                new_height = int(new_width * height / width )
                
                im = im.resize((new_width, new_height), Image.ANTIALIAS)
                width, height = im.size
 
                # Setting the points for cropped image

                
                # Cropped image of above dimension
                # (It will not change original image)
                #im = im.crop((left, top, right, bottom))
                limit = 60

                img.paste(im, (
                    int(6400/2.0 + env.sim.data.qpos[0]*6400/limit), 
                    int(6400/2.0 - env.sim.data.qpos[1]*6400/limit)),
                    im) # env.sim.data.qpos[0]*6400/20) (10-i)*40

            episode_step +=1
            #logger.log('Evaluation/skill'+str(i)+'-speed', prior, episode_step)
        plt.plot(X, Y , label=str(i))
        #w , h = img.size
        #draw = ImageDraw.Draw(img)
        #draw.line([(0, (10-i)*(height)-20), (w,(10-i)*(height)-20)], fill=color[i], width = 10)   
    img.convert('RGB').save(logger._log_dir+"plot.pdf")


    plt.legend(loc='upper left', fontsize=10,) # bbox_to_anchor=(1.2, 1.05),
    #loger.plot(label=i, color=colors_list[i])

    plt.savefig(logger._log_dir+'plotPos.pdf')
    plt.clf()
    plt.close()
    print('***Plot skill***')




import math
import random
from PIL import Image, ImageDraw

def arrowedLine(im, ptA, ptB, width=1, color=(0,255,0)):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    # Get drawing context
    draw = ImageDraw.Draw(im)
    # Draw the line without arrows
    draw.line((ptA,ptB), width=width, fill=color)

    # Now work out the arrowhead
    # = it will be a triangle with one vertex at ptB
    # - it will start at 95% of the length of the line
    # - it will extend 8 pixels either side of the line
    x0, y0 = ptA
    x1, y1 = ptB
    # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
    xb = 0.98*(x1-x0)+x0
    yb = 0.98*(y1-y0)+y0

    # Work out the other two vertices of the triangle
    # Check if line is vertical
    if x0==x1:
       vtx0 = (xb-5, yb)
       vtx1 = (xb+5, yb)
    # Check if line is horizontal
    elif y0==y1:
       vtx0 = (xb, yb+50)
       vtx1 = (xb, yb-50)
    else:
       alpha = math.atan2(y1-y0,x1-x0)-90*math.pi/180
       a = 8*math.cos(alpha)
       b = 8*math.sin(alpha)
       vtx0 = (xb+a, yb+b)
       vtx1 = (xb-a, yb-b)
    print(vtx0, vtx1, ptB)
    #draw.point((xb,yb), fill=(255,0,0))    # DEBUG: draw point of base in red - comment out draw.polygon() below if using this line
    #im.save('DEBUG-base.png')              # DEBUG: save

    # Now draw the arrowhead triangle
    draw.polygon([vtx0, vtx1, ptB], fill=color)
    return im

def plot_skill_pos(agent, env, logger, skill_number, training_config, discriminator, device, timestep=-1,infop=''):
    step_max = training_config['step_max']
    plt.figure(figsize=(6.4, 4.8), dpi=500 )
    img = Image.new("RGBA", (6800, 3300), (255, 255, 255, 1))
    draw = ImageDraw.Draw(img)
    color = ['green', 'red', 'blue', 'orange', 'khaki','cyan', 'purple', 'brown', 'gray', 'olive']
    for i in range (skill_number):
        ## plot variable
        X,Y = [], []
        #Initiate the episode
        obs, prior = env.reset()
        done = False
        episode_step = 0
        #sample skills
        z_vector = np.zeros(skill_number)
        z_vector[i] = 1
        while not(done):
            with eval_mode(agent):
                action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
            obs, reward, done, prior = env.step(action)
            done = True if episode_step + 1 == step_max else False
            #done = True if episode_step == step_max else done
            
            X.append(env.sim.data.qpos[0])
            Y.append(i*10)
            if not(episode_step%7):

                o = env.render(mode='rgb_array')

                h, w = o.shape[:2]
                n = np.dstack((o, np.zeros((h,w),dtype=np.uint8)+255))
                # Make mask of black pixels - mask is True where image is black
                mBlack = (n[:, :, 0:3] == [0,0,0]).all(2)
                #print(mBlack)
                n[mBlack] = (0,0,0,0)
                im = Image.fromarray(n)

                new_width  = 300
                width, height = im.size
                new_height = int(new_width * height / width )
                
                im = im.resize((new_width, new_height), Image.ANTIALIAS)
                width, height = im.size
 
                # Setting the points for cropped image
                left =5
                top = 30
                right = 55
                bottom = height
                
                # Cropped image of above dimension
                # (It will not change original image)
                #im = im.crop((left, top, right, bottom))

                img.paste(im, (int(6000 + env.sim.data.qpos[0]*6000/23), -400+(10-i)*height),im) # env.sim.data.qpos[0]*6400/20) (10-i)*40
            
            episode_step +=1
            logger.log('Evaluation/skill'+str(i)+'-speed', prior, episode_step)
        print(env.sim.data.qpos[0])
        w , h = img.size
        
        draw = ImageDraw.Draw(img)
        draw.line([((w-100),(10-i)*(height)-60), (0, (10-i)*(height)-60)], fill=color[i], width = 10)   
        draw.line([((w-100),(10-i)*(height)-60), (0, (10-i)*(height)-60)], fill=color[i], width = 10) 
    img.convert('RGB').save(logger._log_dir+"plot2.pdf")




def plot_skill_speed(agent, env, logger, skill_number, training_config, discriminator, device, timestep=-1,infop=''):
    step_max = training_config['step_max']
    plt.figure(figsize=(6.4, 4.8), dpi=500 )
   
    for i in range (skill_number):
        ## plot variable
        X,Y = [], []
        #Initiate the episode
        obs, prior = env.reset()
        done = False
        episode_step = 0
        #sample skills
        z_vector = np.zeros(skill_number)
        z_vector[i] = 1
        while not(done):
            with eval_mode(agent):
                action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
            obs, reward, done, prior = env.step(action)
            done = True if episode_step + 1 == step_max else False
            episode_step +=1
            X.append(episode_step)
            Y.append(prior)
        b = dict(skill=torch.tensor(i).long().to(device))
        plt.plot(X, Y, label='Skill '+str(i)+'-'+str(round(discriminator.get_centroids(b).item(),3)), zorder=-1)



    plt.legend(loc='upper left', fontsize=10,) 
    isExist = os.path.exists(logger._log_dir+'/skill'+infop+'/')
    if not isExist:
            os.makedirs(logger._log_dir+'/skill'+infop+'/')
    plt.savefig(logger._log_dir+'/skill'+infop+'/'+str(timestep)+'.pdf')
    plt.clf()
    plt.close()
    print('***Plot skill***')

def plot_agent(agent, env, logger, training_config, timestep=-1,infop=''):
    step_max = training_config['step_max']
    plt.figure(figsize=(6.4, 4.8), dpi=500 )

    ## plot variable
    X,Y = [], []

    #Initiate the episode
    obs = env.reset()
    done = False
    episode_step = 0

    while not(done):
        with eval_mode(agent):
            action = agent.act(obs, sample=True)
        obs, reward, done, info = env.step(action)
        done = True if episode_step + 1 == step_max else False

        episode_step +=1
        X.append(obs[0])
        Y.append(obs[1])
    plt.plot(X, Y )

    plt.scatter(200/255, 200/255, s=40, marker='*', color='y', label='goal-center')
    plt.axis([-1, 1, -1, 1])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.legend(loc='upper left', fontsize=10,)
    isExist = os.path.exists(logger._log_dir+'/skill'+infop+'/')
    if not isExist:
            os.makedirs(logger._log_dir+'/skill'+infop+'/')
    plt.savefig(logger._log_dir+'/skill'+infop+'/'+str(timestep)+'.pdf')
    plt.clf()
    plt.close()
    print('***Plot skill***')