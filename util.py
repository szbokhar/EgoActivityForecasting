import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from pprint import pprint
from numpy.random import choice, rand, randint

from RL_Config import *

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

def make_voxel_grid(pts, colors, block_size, paths=None, alpha=0.2, beta=0.1):
    grid_coords = np.round(pts/block_size, 0).astype(int)

    mx = np.min(grid_coords[:,0], 0)
    mz = np.min(grid_coords[:,1], 0)
    my = np.min(grid_coords[:,2], 0)
    grid_coords[:,0] = grid_coords[:,0] - mx
    grid_coords[:,1] = grid_coords[:,1] - mz
    grid_coords[:,2] = grid_coords[:,2] - my

    if paths is not None:
        for i in range(len(paths)):
            paths[i].smooth_points = smooth_path(paths[i].points, alpha, beta)
            paths[i].block_points = block_path(paths[i].smooth_points, block_size)
            paths[i].points = np.copy(paths[i].block_points)
            paths[i].points[:,0] = paths[i].points[:,0] - mx
            paths[i].points[:,1] = paths[i].points[:,1] - mz
            paths[i].points[:,2] = paths[i].points[:,2] - my

    mx = np.max(grid_coords[:,0], 0)
    mz = np.max(grid_coords[:,1], 0)
    my = np.max(grid_coords[:,2], 0)
    grid = np.zeros((mx+1, mz+1, my+1))

    for i in range(grid_coords.shape[0]):
        p = grid_coords[i,:]
        grid[p[0], p[1], p[2]] += 1

    return grid


def do_qlearn(rl_config, num_iter, rand_count):
    sars = rl_config.total_SARSA_list
    qshape = rl_config.voxel_grid.shape
    alpha = rl_config.alpha
    gamma = rl_config.gamma
    state_size = len(rl_config.rl_state_ids.keys())
    print(rl_config.total_SARSA_list)
    Q = np.zeros(rl_config.q_shape)

    vals = []

    for t in range(num_iter):
        #print('-----', t)
        idx = choice(sars.shape[0], rand_count, replace=False)
        S = sars[idx,:]
        for i in range(len(S)):
            s = S[i, 0:state_size].astype(int)
            act = S[i, state_size]
            ns = S[i, (state_size+2):(2*state_size+2)].astype(int)
            st = tuple(s.tolist() + [act])
            nst = tuple(ns.tolist() + [[x for x in range(Q.shape[-1])]])
            R = S[i, state_size+1]

            Q[st] = Q[st] + alpha*(R + gamma*np.max(Q[nst]) - Q[st])

        if t % 10 == 0:
            vals.append(np.sum(Q))

    return (Q, vals)

def do_explore_qlearn(rl_config, num_iter=2000, rand_count=500, reset_episode=100):
    sars = rl_config.total_SARSA_list
    dplot = rl_config.voxel_grid
    point_sets = rl_config.path_NN
    rl_actions = rl_config.rl_actions
    alpha = rl_config.alpha
    gamma = rl_config.gamma
    epsilon = rl_config.epsilon
    sigma = rl_config.sigma


    # Setup initial Q table
    Q = np.zeros((dplot.shape[0], dplot.shape[2], 3,3,2,2,2, 13))
    umap = np.zeros((dplot.shape[0], dplot.shape[2]))
    pprint(sars)

    # Declare variables for current state and next state
    s = np.array([91, 15, 74, 0, 0, 0, 0, 0])
    ns = np.array([91, 15, 74, 0, 0, 0, 0, 0])

    # for each iteration
    for t in range(num_iter):

        # Reset state after numbre of iterations
        if t % reset_episode == 0:
            print('-----', t)
            s = np.array([randint(dplot.shape[0]), 15, randint(dplot.shape[2]), 0, 0, 0, 0, 0])

        act = -1
        is_greed = rand(1) < epsilon
        if is_greed:
            act = np.argmax(Q[s[0], s[2], s[3], s[4], s[5], s[6], s[7], :])
            val = np.max(Q[s[0], s[2], s[3], s[4], s[5], s[6], s[7], :])
            if val <= 0.00000000001:
                is_greed = False

        if not is_greed:
            if rand(1) < 0.9999:
                act = randint(0, 7)
            else:
                act = randint(7, len(rl_actions))

        ns = np.copy(s)
        if rl_actions[act] is 'Nothing':
            None
        elif rl_actions[act] == 'Move_North':
            ns[2] = min(ns[2]+1, Q.shape[1]-1)
        elif rl_actions[act] == 'Move_South':
            ns[2] = max(ns[2]-1, 0)
        elif rl_actions[act] == 'Move_East':
            ns[0] = min(ns[0]+1, Q.shape[0]-1)
        elif rl_actions[act] == 'Move_West':
            ns[0] = max(ns[0]-1, 0)
        elif rl_actions[act] == 'Move_Up':
            act = 0
        elif rl_actions[act] == 'Move_Down':
            act = 0
        else:
            if (ns[3:8] == np.array([0,0,0,0,0])).all():
                ns[3:8] = np.array([1,0,0,0,0])
                act = 7
            elif (ns[3:8] == np.array([1,0,0,0,0])).all():
                ns[3:8] = np.array([1,0,1,0,0])
                act = 9
            elif (ns[3:8] == np.array([1,0,1,0,0])).all():
                ns[3:8] = np.array([1,1,1,0,0])
                act = 8
            elif (ns[3:8] == np.array([1,1,1,0,0])).all():
                ns[3:8] = np.array([1,1,1,1,0])
                act = 10
            elif (ns[3:8] == np.array([1,1,1,1,0])).all():
                ns[3:8] = np.array([2,1,1,1,0])
                act = 7
            elif (ns[3:8] == np.array([2,1,1,1,0])).all():
                ns[3:8] = np.array([2,1,1,1,1])
                act = 11
            elif (ns[3:8] == np.array([2,1,1,1,1])).all():
                ns[3:8] = np.array([2,2,1,1,1])
                act = 8
            elif (ns[3:8] == np.array([2,2,1,1,1])).all():
                ns[3:8] = np.array([2,2,1,1,1])
                act = 0
        """
        elif rl_actions[act] == 'Do_PickupCup':
            ns[2] = min(ns[2]+1, Q.shape[2]-1)
        elif rl_actions[act] == 'Do_PutdownCup':
            ns[3] = min(ns[3]+1, Q.shape[3]-1)
        elif rl_actions[act] == 'Do_WashCup':
            ns[4] = min(ns[4]+1, Q.shape[4]-1)
        elif rl_actions[act] == 'Do_MakeHotChocolate':
            ns[5] = min(ns[5]+1, Q.shape[5]-1)
        elif rl_actions[act] == 'Do_PickupStraw':
            ns[6] = min(ns[6]+1, Q.shape[6]-1)
        """

        dist = point_sets[tuple(ns[3:8])].query(ns[[0,2]])[0]
        reward = -np.exp(-dist/sigma)
        newstate = np.concatenate((s, np.array([act]), np.array([reward]), ns))
        sars = np.concatenate((sars, newstate.reshape((1,newstate.shape[0]))), 0)
        s = np.copy(ns)

        idx = choice(sars.shape[0], rand_count, replace=False)
        S = sars[idx,:]
        for i in range(len(S)):
            (x,y,uc,dc,wc,hc,us) = (S[i,0], S[i,2], S[i,3], S[i,4], S[i,5], S[i,6], S[i,7])
            act = S[i, 8]
            R = S[i, 9]
            (nx,ny,nuc,ndc,nwc,nhc,nus) = (S[i,10], S[i,12], S[i,13], S[i,14], S[i,15], S[i,16], S[i,17])

            cur = Q[x,y,uc,dc,wc,hc,us,act]

            Q[x,y,uc,dc,wc,hc,us,act] = cur + alpha*(R + gamma*np.max(Q[nx,ny,nuc,ndc,nwc,nhc,nus,:]) - cur)
            umap[x,y] = umap[x,y]+1 #np.max(dplot[x,4:6,y])


    return (Q, umap)

