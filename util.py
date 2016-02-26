import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from pprint import pprint
from numpy.random import choice


def center_points(pts):
    dx = np.sum(pts, 0)/pts.shape[0]
    pts = pts - dx

    return (pts, dx)

def smooth_path(path, alpha, beta):
    spath = np.zeros(path.shape)
    spath[0, :] = path[0, :]
    v = np.array([0.0, 0.0, 0.0])

    for i in range(1, len(path)):
        spath[i, :] = spath[i-1, :] + v
        r = path[i, :] - spath[i, :]
        spath[i, :] = spath[i, :] + alpha * r
        v = v + beta * r

    return spath

def block_path(path, block_size):
    bpath = np.round(path/block_size, 0).astype(int)

    return bpath

def make_voxel_grid(pts, colors, block_size, paths=None):
    grid_coords = np.round(pts/block_size, 0).astype(int)

    mx = np.min(grid_coords[:,0], 0)
    mz = np.min(grid_coords[:,1], 0)
    my = np.min(grid_coords[:,2], 0)
    grid_coords[:,0] = grid_coords[:,0] - mx
    grid_coords[:,1] = grid_coords[:,1] - mz
    grid_coords[:,2] = grid_coords[:,2] - my

    if paths is not None:
        for i in range(len(paths)):
            paths[i] = block_path(smooth_path(paths[i], 0.2, 0.1), block_size)
            paths[i][:,0] = paths[i][:,0] - mx
            paths[i][:,1] = paths[i][:,1] - mz
            paths[i][:,2] = paths[i][:,2] - my

    mx = np.max(grid_coords[:,0], 0)
    mz = np.max(grid_coords[:,1], 0)
    my = np.max(grid_coords[:,2], 0)
    grid = np.zeros((mx+1, mz+1, my+1))

    for i in range(grid_coords.shape[0]):
        p = grid_coords[i,:]
        grid[p[0], p[1], p[2]] += 1

    return grid


def path_data_to_SARS(path, raw_labels, lnames, rl_actions, block_size):
    def get_direction(dx):
        x = dx[0]
        z = dx[1]
        y = dx[2]

        if x == 0 and y == 0 and z == 0:
            return 0
        elif x > 0:
            return 2
        elif x < 0:
            return 4
        elif y > 0:
            return 1
        elif y < 0:
            return 3
        elif z > 0:
            return 5
        elif z < 0:
            return 6

    ka2rl = {3:8, 4:10, 5:9, 6:11, 7:12}
    rl2id = lambda x: x-8
    sa_list = []
    boa = [0, 0, 0, 0, 0]
    nboa = [0, 0, 0, 0, 0]

    for i in range(len(path)-1):
        pos = path[i,:]
        npos = path[i+1,:]
        lbl_num = raw_labels[i]
        lbl = lnames[lbl_num]

        act = -1

        if lbl == 'Standing':
            act = 0     # Nothing
        elif lbl == 'Walking':
            act = get_direction(npos - pos)
        elif lbl_num >= 3:
            nlbl_num = raw_labels[i+1]
            if nlbl_num != lbl_num:
                act = ka2rl[lbl_num]
                nboa[rl2id(act)] += 1
            else:
                act = 0

        state = pos

        state = state.tolist()
        state.extend(boa)
        sa_list.append((state, act))
        boa = list(nboa)

    sars_list = np.zeros((len(sa_list)-1, 18))

    for i in range(len(sa_list)-1):
        sars_list[i, :] = sa_list[i][0] + [sa_list[i][1]] + [0] + sa_list[i+1][0]

    sars_list[-1, 9] = 1

    return sars_list

def do_qlearn(sars, qshape, alpha, num_iter, rand_count):
    Q = np.zeros((qshape[0], qshape[2], 3,3,3,3,3, 13))

    for t in range(num_iter):
        print('-----', t)
        idx = choice(sars.shape[0], rand_count, replace=True)
        S = sars[idx,:]
        for i in range(len(S)):
            x = S[i, 0]
            y = S[i, 2]
            uc = S[i, 3]
            dc = S[i, 4]
            wc = S[i, 5]
            hc = S[i, 6]
            us = S[i, 7]
            act = S[i, 8]
            R = S[i, 9]
            nx = S[i, 10]
            ny = S[i, 12]
            nuc = S[i, 13]
            ndc = S[i, 14]
            nwc = S[i, 15]
            nhc = S[i, 16]
            nus = S[i, 17]

            cur = Q[x,y,uc,dc,wc,hc,us,act]

            Q[x,y,uc,dc,wc,hc,us,act] = cur + alpha*(R + np.max(Q[nx,ny,nuc,ndc,nwc,nhc,nus,:]) - cur)

    return Q

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

    start = 50
    for i in range(25):
        a = fig.add_subplot(5,5,i+1)
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

    plt.show()
