import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from os import listdir
from os.path import isfile, join
import os
from glob import glob
import matplotlib.pyplot as plt
import itertools, os
import torch
import numpy as np
import matplotlib as mpl

def get_dataframe(path):
        runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
        #path = './Projects/Project1N/experiments/nav2d/test/cdp/results-t3/tb/events.out.tfevents.1664408925.4gpu-05.1438093.0'
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
        return runlog_data
    #runlog_data = get_dataframe
def get_files(path, method):
    files = []
    pattern   =  "*/tb/*"+ method

    for dir,_,_ in os.walk(path):
        files.extend(glob(os.path.join(dir,pattern))) 
    print(files)
    return files


def plot_lib(labels, color, soft_color, means, stds, timesteps, file_save, x_label, y_label):
    
    
    plt.figure(figsize=(10.4, 6.8), dpi=500 )
    #Edl
    i = 0 
    for label, value, std in zip(labels,  means, stds):
        plt.plot(timesteps, value, color[i%3], label=label) #s=10,c=(i+1)/101, cmap='Greys', marker='+')
        plt.fill_between(timesteps, value-std, value+std, color=soft_color[i%3],alpha=0.3)
        i+=1

    plt.xlabel(x_label,fontsize=18)
    plt.ylabel(y_label ,fontsize=18)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               mode="expand", borderaxespad=0., ncol=5, fontsize='xx-large') #labelspacing=1,  fontsize=20)
    plt.grid()
    plt.savefig(file_save)
    plt.clf()

def plot(path, methods):
    dic = dict()
    for method in methods:
        dic['method'] = [get_dataframe(path) for path in get_files(path, method)]
    

    ## Plot reward
    x = dic['cdp'][0][dic['cdp'][0].metric.str.endswith("/reward")].step


def plot_beta(path):
    methods = ['0.1','0.2' ,'0.3','0.4','0.5','0.6','0.7','0.8','0.9']
    color = ['green', 'red', 'blue']
    labels=methods
    dic = dict()
    print(path)
    for method in methods:
        dic[method] = [get_dataframe(path) for path in get_files(path, method)]

    timestep = dic['0.1'][0][dic['0.1'][0].metric.str.endswith("episode_reward")].step.to_numpy()
    means=[]
    stds=[]
    for method in methods:
        r = np.array([s[s.metric.str.endswith("episode_reward")].value.to_numpy() for s in dic[method]])
        means.append(smooth2(np.mean(r, axis=0)))
        stds.append(smooth2(np.std(r, axis=0)))
    plot_lib(labels,color ,  means, stds, timestep,path+'reward.pdf','Number of steps', 'Average return')


def smooth(values, average_time):
    return np.mean(np.split(values, average_time),axis=0)

def smooth2(values):
    t = []
    for i, value in enumerate(values):

        t.append(np.mean(values[:i+1]))
    return np.array(t)

def plot_gep(path):
    smoothing_t = 2
    methods = ['smm','smm_prior','cdp']
    dic = dict()
    for method in methods:
        dic[method] = [get_dataframe(path) for path in get_files(path, method)]
    ## Reward
    labels = ['SMM', 'SMM+prior', 'CDP (ours)']

    means = []
    stds = []

    timestep = dic['smm'][0][dic['smm'][0].metric.str.endswith("episode_reward")].step.to_numpy()

    color = ['green', 'red', 'blue']
    soft_color = ['lightgreen', 'darksalmon', 'lightskyblue']
    for method in methods:
        r = np.array([s[s.metric.str.endswith("episode_reward")].value.to_numpy() for s in dic[method]])
        means.append(smooth2(np.mean(r, axis=0)))
        stds.append(smooth2(np.std(r, axis=0)))
    plot_lib(labels, color, soft_color ,  means, stds, timestep, path+'reward.pdf','Number of steps', 'Average return')


def plot_speed(path):
    isExist = os.path.exists(path+'/speed/')
    if not isExist:
            os.makedirs(path+'/speed/')
    smoothing_t = 2
    methods = ['0.3', '0.5','0.9']
    dic = dict()
    color = ['green', 'red', 'blue', 'orange', 'y','c', 'purple', 'brown', 'gray', 'olive']
    for method in methods:
        dic[method] = [get_dataframe(path) for path in get_files(path, method)]
    ## Reward
    plt.figure(figsize=(6, 4.8), dpi=500 )
    timesteps = dic['0.3'][0][dic['0.3'][0].metric.str.endswith("0-speed")].step.to_numpy()
    std = []
    
    for method in methods:
        r = []
        for i in range(10):
            speeds = np.array([s[s.metric.str.endswith(str(i)+"-speed")].value.to_numpy() for s in dic[method]])

            smooths = smooth2(speeds[0])
            r.append(smooths)
        r = np.array(r)
        for i in range(10):
            plt.plot(timesteps, r[i], color[i], label="Skill " + str(1+i)) #s=10,c=(i+1)/101, cmap='Greys', marker='+')

        plt.xlabel('Number of steps',fontsize=18)
        plt.ylabel("HC's velocity" ,fontsize=18)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               mode="expand", borderaxespad=0., ncol=5, fontsize='xx-large')
        plt.axis([0, 100, -4, 1])
        plt.grid()
        plt.savefig(path+'/speed/'+method+'sp.pdf')
        plt.clf()
        #r = np.mean(r,axis=-1)
        rs = np.var(r,axis=0)
        std.append(rs)

    for i, method in enumerate(methods):
            plt.plot(timesteps, std[i], label='$\u03B2$='+method) #s=10,c=(i+1)/101, cmap='Greys', marker='+')
    plt.xlabel('Number of steps',fontsize=18)
    plt.ylabel("Variance between skills's velocity" ,fontsize=16)
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               mode="expand", borderaxespad=0., ncol=5, fontsize='xx-large')
    #plt.legend()
    plt.grid()
    plt.savefig(path+'/speed/beta.pdf')
    plt.clf()
    plt.close()

def main(args):
    path = args.dir
   
    if args.plot =='beta':
        plot_beta(path)
    if args.plot =='gep':
        plot_gep(path)
    if args.plot =='speed':
        plot_speed(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('-plot', default=None)
    parser.add_argument('-dir', default=None)
    args = parser.parse_args()



    main(args)