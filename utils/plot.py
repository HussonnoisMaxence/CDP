
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_env
import utils.utils as utils
from utils.networks import eval_mode
import torch
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection
import numpy as np
import matplotlib as mpl
from PIL import Image, ImageDraw
from gym.wrappers import RecordVideo



clist = ['b','orange', 'purple', 'y', 'r', 'pink', 'green', 'brown', 'cyan', 'violet', ' chocolat', 'teal']
color = ['green', 'red', 'blue', 'orange', 'sienna', 'blueviolet', 'teal', 'grey']
soft_color = ['lightgreen', 'darksalmon', 'lightskyblue', 'gold', 'sandybrown', 'plum', 'lightblue', 'lightgrey']

def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
def update(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([100])


## Plot related to the discriminator
def plot_discr_region_pref(device, reward_model, discriminator, info):
    obs = np.array(utils.get_pairs(-1, 1, 0.005) )
    obs_tensor = torch.tensor(obs, dtype=torch.float).to(device)
    dataset = reward_model.f_hat_batch(obs_tensor, tensor=True)

    x = obs[:,0]
    y = obs[:,1]
    obs = [[o]for o in obs]
    vmin = -5
    vmax = 0    
    plt.figure(figsize=(10.4, 8.8), dpi=500 )
    color_mapping = []
    with torch.no_grad():
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
            s = discriminator.get_centroids(dict(skill=torch.tensor(emb_idx).to(device)))[0] #.detach().cpu().numpy()

            b = s.repeat((dataset.shape[0],1))

            b = torch.square(b-dataset).mean(dim=-1)

            s = obs_tensor[torch.argmin(b)].detach().cpu().numpy()

            #print(s)
            m = plt.scatter(s[0], s[1],  edgecolor='black', linewidth=0.7,linestyle=':',  label='Skill {}'.format(emb_idx+1))
            c.append(s)
    plt.axis([-1, 1, -1, 1])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    

    dir = f"{info['dir']}/discriminator/region"
    utils.create_directories(dir)
    if info['get_legends']:
        lgnd = plt.legend(
             bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   mode="expand", borderaxespad=0., ncol=3, fontsize='xx-large', scatterpoints=1,
             handler_map={PathCollection : HandlerPathCollection(update_func=update)}) 

        export_legend(lgnd, filename=f"{dir}/legend.pdf")


    
    if info['pdf']:
          plt.savefig(f"{dir}/{info['timestep']}.pdf")

    
    plt.savefig(f"{dir}/{info['timestep']}.png")
    plt.clf()
    plt.close()
    return c, color_mapping

def plot_discr_region(device, discriminator, info):
    obs = utils.get_pairs(-1, 1, 0.005) 
    obs = torch.tensor(np.array(obs), dtype=torch.float)
    dataset = obs.to(device)
    obs = obs.detach().cpu().numpy()
    x = obs[:,0]
    y = obs[:,1]
    obs = [[o]for o in obs]
    vmin = -5
    vmax = 0    
    plt.figure(figsize=(10.4, 8.8), dpi=500 )
    color_mapping = []
    with torch.no_grad():
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
    

    dir = f"{info['dir']}/discriminator/region"
    utils.create_directories(dir)
    if info['get_legends']:
        lgnd = plt.legend(
             bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   mode="expand", borderaxespad=0., ncol=4, fontsize='xx-large', scatterpoints=1,
             handler_map={PathCollection : HandlerPathCollection(update_func=update)}) 

        export_legend(lgnd, filename=f"{dir}/legend.pdf")


    
    if info['pdf']:
          plt.savefig(f"{dir}/{info['timestep']}.pdf")

    
    plt.savefig(f"{dir}/{info['timestep']}.png")
    plt.clf()
    plt.close()
    return c, color_mapping


## Plot related to the rewards
def plot_reward_nc(rewards, obs,  info):

    x = obs[:,0]
    y = obs[:,1]
    obs = [[o]for o in obs]

    vmin = -1
    vmax = 1
    plt.figure(figsize=(10.4, 8.8), dpi=500 )
    ax = plt.gca()

    plt.scatter(x, y, s=3, marker='o', c=rewards, vmin=vmin, vmax=vmax, cmap='bwr')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    #ax.set_title('Reward from Human Preferences', fontsize=14)
    cbar = plt.colorbar(cmap='bwr', orientation="vertical", pad=0.025)
    cbar.set_label('Reward value', fontsize='xx-large')
    dir =f"{info['dir']}/rewards"
    utils.create_directories(dir)
    if info['get_legends']:
        legend = plt.legend(
             bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', 
             mode="expand", borderaxespad=0., ncol=3, fontsize='xx-large', scatterpoints=1,
             handler_map={PathCollection : HandlerPathCollection(update_func=update)}) 
        export_legend(legend, filename=f"{dir}/legend_{info['timestep']}.pdf")

    plt.savefig(f"{dir}/{info['timestep']}.png")
    if info['pdf']:
          plt.savefig(f"{dir}/{info['timestep']}.pdf")

    
    plt.clf()
    plt.close()





## Plot related to exploration
def plot_state(datas, info, area=None):
    plt.figure(figsize=(10.4, 8.8), dpi=500 )
    ax = plt.gca()
    area = info['area']
    if not(area == None):
        if len(area)==4:
            ax.add_patch(
                 plt.Rectangle(
                      [area[0],area[1]],
                      area[2]-area[0],area[3]-area[1],
                      facecolor='y', 
                      fill=False,
                      label='Target Region',
                      edgecolor='y',
                      linewidth=2.5,
                      zorder=1))
        if len(area)==2:
             ax.plot(area[0], area[1], marker="*", color='y', label='Target Region Center')

    for label, data in enumerate(datas):
        obs = data
        obs = obs
        x = obs[:,0]
        y = obs[:,1]
        plt.scatter(x, y, s=1,label='Preferred Region', zorder=-1)



  
    
    plt.scatter(0, 0, s=60, marker='+', color='r', label='Initial State')
    if info['env'] != 'Ant':
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    dir =f"{info['dir']}/coverage/{info['prefix']}"
    utils.create_directories(dir)

    if info['get_legends']:
        lgnd = plt.legend(
             bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', 
             mode="expand", borderaxespad=0., ncol=3, fontsize='xx-large', scatterpoints=1,
             handler_map={PathCollection : HandlerPathCollection(update_func=update)}) 

        export_legend(lgnd, filename=f"{dir}/legend_{info['prefix']}.pdf")


    plt.savefig(f"{dir}/{info['timestep']}.png")
    if info['pdf']:
          plt.savefig(f"{dir}/{info['timestep']}.pdf")

    
    plt.clf()
    plt.close()
    
def plot_state_visited(data, info):

    plt.figure(figsize=(10.4, 8.8), dpi=500 )

    reds = plt.get_cmap("Reds")
    norm = mpl.colors.Normalize(vmin=0, vmax=101)
    #cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)

    #size = x.shape[0]//4
    #t = 1
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap='YlOrRd')
    for index, traj in enumerate(data):
        if not(index%3):
            for i, el in enumerate(traj[0]):
                if not(i%8):
                    x = el[0]
                    y = el[1]
                    plt.scatter(x, y, s=0.7, color=cmap.to_rgba(i+1))
    ax = plt.gca()
    area = info['area']
    ax.add_patch(
        plt.Rectangle(
        [area[0],area[1]],
        area[2]-area[0],area[3]-area[1],
        facecolor='y', 
        fill=False,
        label='Target Region',
        edgecolor='y',
        linewidth=2.5,
        zorder=1))
    
    plt.axis([-1, 1, -1, 1])
    cbar = plt.colorbar(cmap, orientation = 'horizontal')
    cbar.set_label('Steps', fontsize='xx-large')

    ax = plt.gca()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)



    dir =f"{info['dir']}/exploration/state_visited"
    utils.create_directories(dir)
    if info['get_legends']:
        legend = plt.legend(
             bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', 
             mode="expand", borderaxespad=0., ncol=3, fontsize='xx-large', scatterpoints=1,
             handler_map={PathCollection : HandlerPathCollection(update_func=update)}) 
        export_legend(legend, filename=f"{dir}/legend_{info['timestep']}.pdf")
    plt.savefig(f"{dir}/states_visited_{info['timestep']}.png")
    if info['pdf']:
        plt.savefig(f"{dir}/states_visited_{info['timestep']}.pdf")
    plt.clf()
    plt.close()

def plot_state_hc(datas, info, area=None):
    plt.figure(figsize=(10.4, 8.8), dpi=500 )
    ax = plt.gca()
    area = info['area']
    if not(area == None):
        if len(area)==2:
            ax.axvline(x = area[0], color='y', label='Target Region')
            ax.axvline(x = area[1], color='y', label='Target Region2')
    for label, data in enumerate(datas):
        obs = data
        obs = obs
        x = obs
        y = np.zeros_like(obs)
        plt.scatter(x, y, s=1,label='Preferred Region', zorder=-1)



  
    
    plt.scatter(0, 0, s=60, marker='+', color='r', label='Initial State')



    dir =f"{info['dir']}/coverage/{info['prefix']}"
    utils.create_directories(dir)

    if info['get_legends']:
        lgnd = plt.legend(
             bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', 
             mode="expand", borderaxespad=0., ncol=3, fontsize='xx-large', scatterpoints=1,
             handler_map={PathCollection : HandlerPathCollection(update_func=update)}) 

        export_legend(lgnd, filename=f"{dir}/legend_{info['prefix']}.pdf")


    plt.savefig(f"{dir}/{info['timestep']}.png")
    if info['pdf']:
          plt.savefig(f"{dir}/{info['timestep']}.pdf")

    
    plt.clf()
    plt.close()

## Plot related to trajectory
def plot_traj_hc(data):

    plt.figure(figsize=(10.4, 8.8), dpi=500 )

    if data['color_mapping'] == None:
        data['color_mapping'] = clist[:len(data['traj'])]
        
    for i, data_traj in enumerate(data['traj']):
        traj= np.array(data_traj)
        X, Y = traj, np.ones_like(traj)*(i+2)
 
        plt.plot(X, Y, label='Skill '+str(1+i), c=data['color_mapping'][i], linewidth=1 ,zorder=-1)

     
    dir =f"{data['dir']}/{data['phase']}/Skills"
    utils.create_directories(dir)
    if data['get_legends']:
        legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                mode="expand", borderaxespad=0., ncol=4, fontsize='xx-large')
        export_legend(legend, filename=f"{dir}/legend_skills_{data['timestep']}.pdf")
    plt.savefig(f"{dir}/skills_{data['timestep']}.png")
    if data['pdf']:
        plt.savefig(f"{dir}/skills_{data['timestep']}.pdf")
    plt.clf()
    plt.close()

def plot_traj(data):

    plt.figure(figsize=(10.4, 8.8), dpi=500 )

    if data['color_mapping'] == None:
        data['color_mapping'] = clist[:len(data['traj'])]

    for i, data_traj in enumerate(data['traj']):
        traj= np.array(data_traj)
        X, Y = traj[:,0], traj[:,1]

        plt.plot(X, Y, label='Skill '+str(1+i), c=data['color_mapping'][i], linewidth=1 ,zorder=-1)
        plt.scatter(data['centroids'][i][0], data['centroids'][i][1], s=160, marker="*", edgecolor='black', c=data['color_mapping'][i], linewidth=0.7,zorder=1)
        #ax.scatter(X[-1], Y[-1],marker='*',s=80, color='k')
        
    if data['env'] != 'Ant':
        ax = plt.gca()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        if data['plot_area']:
            area = data['area']    
            ax.add_patch(
                plt.Rectangle(
                    [area[0],area[1]],
                    area[2]-area[0],area[3]-area[1],
                    facecolor='y', 
                    fill=False,
                    #label='Target Region',
                    edgecolor='y',
                    linewidth=2.5,
                    zorder=1))
     
    dir =f"{data['dir']}/{data['phase']}/Skills"
    utils.create_directories(dir)
    if data['get_legends']:
        legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                mode="expand", borderaxespad=0., ncol=4, fontsize='xx-large', scatterpoints=1,
             handler_map={PathCollection : HandlerPathCollection(update_func=update)})
        export_legend(legend, filename=f"{dir}/legend_skills_{data['timestep']}.pdf")
    plt.savefig(f"{dir}/skills_{data['timestep']}.png")
    if data['pdf']:
        plt.savefig(f"{dir}/skills_{data['timestep']}.pdf")
    plt.clf()
    plt.close()

def save_frames(info):
    dir =f"{info['dir']}/{info['phase']}/Skills_record/{info['timestep']}"
    utils.create_directories(dir)
    for z, traj in enumerate(info['frames']):
        imgs = [Image.fromarray(img) for img in traj]
        # duration is the number of milliseconds between frames; this is 40 frames per second
        imgs[0].save(f"{dir}/Skill_{z}.gif", save_all=True, append_images=imgs[1:], duration=50, loop=1)

def plot_velocity(datas):
    x_label = "# of steps"
    y_label = "F1 Score"
    plt.figure(figsize=(10.4, 8.8), dpi=500 )
    ax = plt.gca()
    x = np.arange(0,100)
    for index, velocity in enumerate(datas['velocities']):
        plt.plot(x,velocity, label=f"Skill {index}") #s=10,c=(i+1)/101, cmap='Greys', marker='+')

    plt.xlabel(x_label,fontsize=27)
    plt.ylabel(y_label ,fontsize=27)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax = plt.gca()
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    

    legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    mode="expand", borderaxespad=0., ncol=len(datas['velocities']), fontsize='x-large', scatterpoints=1,



                handler_map={PathCollection : HandlerPathCollection(update_func=update)}) 
    dir =f"{datas['dir']}/skill_velocity"
    utils.create_directories(dir)
    export_legend(legend, filename=f"{dir}/legend.pdf")
    legend.remove()
        #labelspacing=1,  fontsize=20)
        
        
    plt.grid()
    plt.savefig(f"{dir}/plot.png")
    if datas['pdf']:
          plt.savefig(f"{dir}/plot.pdf")
    plt.clf()

    plt.close()


def plot_pos(datas):
    img = Image.new("RGBA", (6800, 3300), (255, 255, 255, 1))
    draw = ImageDraw.Draw(img)
    color = ['green', 'red', 'blue', 'orange', 'khaki','cyan', 'purple', 'brown', 'gray', 'olive']
    for z, traj in enumerate(datas['frames']):
        for t, o in enumerate (traj):
            if not(t%7):
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
                x_pos = datas['traj'][z][t]
                img.paste(im, (int(6000 + x_pos*6000/23), -400+(10-z)*height),im)
        w , h = img.size
        draw = ImageDraw.Draw(img)
        draw.line([((w-100),(10-z)*(height)-60), (0, (10-z)*(height)-60)], fill=color[z], width = 10)   
        draw.line([((w-100),(10-z)*(height)-60), (0, (10-z)*(height)-60)], fill=color[z], width = 10) 
    dir =f"{datas['dir']}/skill_position/{datas['timestep']}"
    utils.create_directories(dir)
    img.convert('RGB').save(f"{dir}/skill_pos.pdf")
    

## Other
def save_data(data, info):
    dir = f"{info['dir']}/exploration/state_visited"
    utils.create_directories(dir)
    print(dir)
    np.save(arr=data, file=f"{dir}/states_visited")

def get_color_mapping(use_vqvae, discovery_div_embedding, device, reward_model, discriminator, settings_info):
    if discovery_div_embedding == 'reward_feature':
        c, color_mapping = plot_discr_region_pref(device, reward_model, discriminator, settings_info)
    else:
        if use_vqvae:
            c, color_mapping = plot_discr_region(device, discriminator, settings_info)
        else:
            color_mapping = None
            c=None
    return color_mapping, c