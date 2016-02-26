import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from numpy.random import choice

def show_action_value(Q):
    fig = plt.figure(1)

    vmin = np.min(Q)
    vmax = np.max(Q)
    cmap_name = 'inferno'

    a = fig.add_subplot(3,3,1)
    plt.imshow(np.max(Q[:,:,0,0,0,0,0,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
    a.set_title('0,0,0,0,0')

    a = fig.add_subplot(3,3,2)
    plt.imshow(np.max(Q[:,:,1,0,0,0,0,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
    a.set_title('1,0,0,0,0')

    a = fig.add_subplot(3,3,3)
    plt.imshow(np.max(Q[:,:,1,0,1,0,0,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
    a.set_title('1,0,1,0,0')

    a = fig.add_subplot(3,3,4)
    plt.imshow(np.max(Q[:,:,1,1,1,0,0,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
    a.set_title('1,1,1,0,0')

    a = fig.add_subplot(3,3,5)
    plt.imshow(np.max(Q[:,:,1,1,1,1,0,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
    a.set_title('1,1,1,1,0')

    a = fig.add_subplot(3,3,6)
    plt.imshow(np.max(Q[:,:,2,1,1,1,0,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
    a.set_title('2,1,1,1,0')

    a = fig.add_subplot(3,3,7)
    plt.imshow(np.max(Q[:,:,2,1,1,1,1,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
    a.set_title('2,1,1,1,1')

    a = fig.add_subplot(3,3,8)
    plt.imshow(np.max(Q[:,:,2,2,1,1,1,:], 2), cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
    a.set_title('2,2,1,1,1')

    plt.tight_layout()



def show_grid(grid):
    fig = plt.figure(2)

    vmin = np.min(grid)
    vmax = np.max(grid)/8
    cmap_name = 'inferno'

    print(grid.shape, vmin, vmax)

    for i in range(grid.shape[1]):
        print(i, np.max(grid[:,i,:]))

    start = 3
    for i in range(30):
        a = fig.add_subplot(5,6,i+1)
        plt.imshow(grid[:,start+i,:], cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
        a.set_title(str(start+i))



def make_basic_plot(pts, colors, path, num):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[0:num, 0], pts[0:num, 1], pts[0:num, 2], s=3, c=colors[0:num], edgecolors='none', zdir='y')

    ax.plot([l[0] for l in path], [l[1] for l in path], [l[2] for l in path], zdir='y')

    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])
    axes.set_zlim([-3, 3])
    axes.invert_zaxis()
