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
        idx = choice(sars.shape[0], rand_count, replace=True)
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
    qshape = rl_config.voxel_grid.shape

    alpha = rl_config.alpha
    gamma = rl_config.gamma
    epsilon = rl_config.epsilon
    sigma = rl_config.sigma

    state_size = len(rl_config.rl_state_ids.keys())
    rid2rl_actions = rl_config.rl_actions
    rl_actions2rid = {v: k for k, v in rid2rl_actions.items()}

    print(rl_config.total_SARSA_list)
    Q = np.zeros(rl_config.q_shape)
    umap = np.zeros(rl_config.q_shape)

    vals = []

    e_state = np.zeros(state_size)
    next_e_state = np.zeros(state_size)

    e_length = 0
    for t in range(num_iter):
        if e_length % reset_episode == 0:
            e_state = rl_config.get_random_state(rl_config)
            print("Step:", t)

        (next_e_state, new_sarsa_state, isFinished) = rl_config.explore_step(rl_config, Q, e_state, epsilon=epsilon)
        sars = np.concatenate((sars, new_sarsa_state), axis=0)

        e_state = np.copy(next_e_state)
        if isFinished:
            e_length = 0
        else:
            e_length += 1

        #print(new_sarsa_state)
        #print('-----', t)
        idx = choice(sars.shape[0], rand_count, replace=True)
        S = sars[idx,:]
        for i in range(len(S)):
            s = S[i, 0:state_size].astype(int)
            act = S[i, state_size]
            ns = S[i, (state_size+2):(2*state_size+2)].astype(int)
            st = tuple(s.tolist() + [act])
            nst = tuple(ns.tolist() + [[x for x in range(Q.shape[-1])]])
            R = S[i, state_size+1]

            Q[st] = Q[st] + alpha*(R + gamma*np.max(Q[nst]) - Q[st])
            umap[st] += 1

        if t % 10 == 0:
            # print(vals[-1] if len(vals) > 1 else None)
            vals.append(np.sum(Q))


    return (Q, vals, umap)
