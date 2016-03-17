import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from numpy.random import choice

from RL_Config import *

def show_value(Q, p):
    fig = plt.figure(p)

    print(Q.shape)
    vmin = None#np.min(Q)
    vmax = None#np.max(Q)
    print('min', vmin, 'max', vmax)
    cmap_name = 'inferno'

    a = fig.add_subplot(1,3,1)
    plt.imshow(np.max(Q[:,:,0,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax, interpolation='nearest')
    a.set_title('0')

    a = fig.add_subplot(1,3,2)
    plt.imshow(np.max(Q[:,:,1,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax, interpolation='nearest')
    a.set_title('1')

    a = fig.add_subplot(1,3,3)
    plt.imshow(np.max(Q[:,:,2,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax, interpolation='nearest')
    a.set_title('2')

    plt.tight_layout()

def plot_1D(data):
    fig = plt.figure(2)
    plt.plot(data, 'r-')

def show_action_value(Q, plot, s):
    fig = plt.figure(plot)

    vmin = None#np.min(Q[:,:,s[0],:])
    vmax = None#np.max(Q[:,:,s[0],:])
    print('min', vmin, 'max', vmax)
    cmap_name = 'jet'

    SX = Q.shape[0]
    SY = Q.shape[1]
    SA = Q.shape[-1]
    for i in range(SA):
        a = fig.add_subplot(4,3,i+1)
        idx = [[x for x in range(SX)]] + [[x for x in range(SY)]] + s + [i]
        plt.imshow(Q[:,:,s[0],i], cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax, interpolation='nearest')
        a.set_title(str(i))

    plt.tight_layout()



def show_grid(grid, start, maxdiv):
    fig = plt.figure(3)

    vmin = None#np.min(grid)
    vmax = None#np.max(grid)/maxdiv
    cmap_name = 'inferno'

    print(grid.shape, vmin, vmax)

    for i in range(grid.shape[1]):
        print(i, np.max(grid[:,i,:]))

    for i in range(9):
        a = fig.add_subplot(2,5,i+1)
        plt.imshow(grid[:,start+i,:], cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax, interpolation='nearest')
        a.set_title(str(start+i))

    fig = plt.figure(30)
    a = fig.add_subplot(1,1,1)
    tmp = 10*np.max(grid[:,6:12,:], axis=1)/1081
    plt.imshow(tmp, cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax, interpolation='nearest')
    a.set_title('cost')



def make_basic_plot(rl_config, pid, path_colors, num):
    pts = rl_config.pc_points
    colors = rl_config.pc_colors
    paths = [rl_config.paths[pid].block_points*rl_config.blocksize, rl_config.paths[pid].smooth_points, rl_config.paths[pid].raw_points]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[0:num, 0], pts[0:num, 1], pts[0:num, 2], s=3, c=colors[0:num], edgecolors='none', zdir='y')

    for (p,c) in zip(paths, path_colors):
        print(p)
        ax.plot([l[0] for l in p], [l[1] for l in p], [l[2] for l in p], c, zdir='y')

    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])
    axes.set_zlim([-3, 3])
    axes.invert_zaxis()

def plot_path_reward(point_sets, dplot, sigma=0.1):
    w = dplot.shape[0]
    h = dplot.shape[2]

    fig = plt.figure(4)

    i=0
    print(point_sets)
    for k in point_sets:
        print(k)
        mymap = np.zeros((len(point_sets), w, h))
        for x in range(w):
            for y in range(h):
                rew = point_sets[k].query(np.array([x,y]))
                mymap[i,x,y] = np.exp(-rew[0]/sigma)


        vmin = np.min(mymap[i,:,:])
        vmax = np.max(mymap[i,:,:])
        a = fig.add_subplot(2,4,i+1)
        plt.imshow(mymap[i,:,:], cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax, interpolation='nearest')
        a.set_title(str(k))
        i=i+1

    plt.tight_layout()

    
