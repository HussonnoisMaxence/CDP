import argparse
import matplotlib.pyplot as plt
import itertools, os
import torch
import numpy as np
import matplotlib as mpl

import matplotlib.patches as mpatches



def plot_exploration(file, path_to_save):
    
    x = np.load(file, allow_pickle=True)
    x = x[:500,:]
    reds = plt.get_cmap("Reds")
    plt.figure(figsize=(6.4, 4.8), dpi=200 )
    norm = mpl.colors.Normalize(vmin=0, vmax=101)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    clist = ['YlOrBr', 'Greys', 'Purples', 'Greens', 'Oranges', 'Reds' ,'spring',
                         'OrRd', 'PuRd', 'RdPu', 'BuPu',
                        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    size = x.shape[0]//4
    t = 1
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap='YlOrRd')
    for index, data in enumerate(x):
        if (index+1)%(size*t)==0:
            t = t+1
        print (index, t, (index+1)%(size*t))
        for i, el in enumerate(data[0]):
            if not(i%8):
                x = el[0]
                y = el[1]

                plt.scatter(x, y,s=0.7, color=cmap.to_rgba(i+1))

    plt.axis([-1, 1, -1, 1])
    cbar = plt.colorbar(cmap)
    cbar.set_label('Steps', fontsize='xx-large')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(path_to_save+'.pdf')
    plt.clf()
    plt.close()

def main(args):
    path = args.dir
    path_to_save = path + '/PlotExploration/'
    algos = ['smm','smm_prior', 'cdp'] 
    seeds = [12345, 23451, 34512, 51234, 67890, 78906, 89067]
    for algo in algos:
        for seed in seeds:
            file =  path  +algo  +'/' + str(seed) +'/data/exploration.npy'
            path_to_save = path_to_save + '/' + algo + '/' + str(seed)
            isExist = os.path.exists(path_to_save)
            if not isExist:
                os.makedirs(path_to_save)
            plot_exploration(file, path_to_save + '/' +str(seed))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('-dir', default=None)
    args = parser.parse_args()



    main(args)
#plot_reward()path_to_save = path + '/' + algo + '/' +str(seed)