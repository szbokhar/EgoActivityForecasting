import sys
import argh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import ipdb
import os

import util
import load_data
import sarsa_util
import display
from RL_Config import *


@argh.arg('points_file',
        help='File containing point cloud data as list of points')
@argh.arg('path_pat',
        help='Filename pattern for path file data (eg. data/qm_hc{0}_{1}.txt)')
@argh.arg('data_ids', help='List of data ids', nargs='+', type=int)
@argh.arg('config_dir', help='Config directory')
@argh.arg('-b', '--blocksize', help='Grid block size', default=0.5)
@argh.arg('-s', '--sigma', help='Path reward sigma', default=5000)
def plot_path_rewards(points_file, path_pat, data_ids, config_dir, **extra):
    "Run basic q-learning algorithm"
    rl_config = RL_Config()
    rl_config.set_parameters(
            blocksize=extra['blocksize'])
    rl_config.load_pointcloud(points_file)
    rl_config.load_action_files(config_dir)
    rl_config.load_path_data(path_pat, data_ids)
    rl_config.format_grid_and_paths()

    display.plot_path_reward(rl_config.path_NN, rl_config.voxel_grid, extra['sigma'])

    plt.show()

@argh.arg('points_file',
        help='File containing point cloud data as list of points')
@argh.arg('path_pat',
        help='Filename pattern for path file data (eg. data/qm_hc{0}_{1}.txt)')
@argh.arg('data_ids', help='List of data ids', nargs='+', type=int)
@argh.arg('-b', '--blocksize', default=0.5, help='Side length of grid cube')
@argh.arg('-s', '--start', default=0, help='Z level to begin plot at')
@argh.arg('-m', '--max_div', default=8, help='Divide max by this')
def show_denseplot(points_file, path_pat, data_ids, **extra):
    "Generate and show pointclound density plot"

    rl_config = RL_Config()
    rl_config.set_parameters(blocksize=extra['blocksize'])
    rl_config.load_pointcloud(points_file)
    rl_config.load_path_data(path_pat, data_ids)
    rl_config.format_grid_and_paths()

    display.show_grid(rl_config.voxel_grid, extra['start'], rl_config.person_column, extra['max_div'])

    plt.show()

@argh.arg('points_file',
        help='File containing point cloud data as list of points')
@argh.arg('path_pat',
        help='Filename pattern for path file data (eg. data/qm_hc{0}_{1}.txt)')
@argh.arg('data_ids', help='List of data ids', nargs='+', type=int)
@argh.arg('-c', '--count', default=4000, help='Number of points to plot')
@argh.arg('-b', '--blocksize', default=0.5, help='Side length of grid cube')
def show_points_and_path(points_file, path_pat, data_ids, **extra):
    """
        Loads points and path data files and plots them
    """

    count = extra['count']
    rl_config = RL_Config()
    rl_config.set_parameters(blocksize=extra['blocksize'])
    rl_config.load_pointcloud(points_file)
    rl_config.load_path_data(path_pat, data_ids)
    rl_config.format_grid_and_paths()

    display.make_basic_plot(rl_config, 0, ['b-', 'r-', 'g-'], count)

    plt.show()


@argh.arg('points_file',
        help='File containing point cloud data as list of points')
@argh.arg('path_pat',
        help='Filename pattern for path file data (eg. data/qm_hc{0}_{1}.txt)')
@argh.arg('data_ids', help='List of data ids', nargs='+', type=int)
@argh.arg('config_dir', help='Config directory')
@argh.arg('-a', '--alpha', help='Learning rate', default=0.5)
@argh.arg('-g', '--gamma', help='Discount factor', default=0.5)
@argh.arg('-b', '--blocksize', help='Grid block size', default=0.5)
@argh.arg('-i', '--iter', help='Number of q-learning iterations', default=1000)
@argh.arg('-m', '--memory_size', help='Iteration sample size', default=200)
@argh.arg('--state_functions', help='Functions specification',
        default=['hc_only_make_sarsa_lists','hc_only_NN','hc_only_reward','hc_only_transition'], nargs='+', type=str)
def basic_qlearn(points_file, path_pat, data_ids, config_dir, **extra):
    "Run basic q-learning algorithm"
    num_iter = extra['iter']
    memory_size = extra['memory_size']
    training_paths = []
    training_labels = []

    rl_config = RL_Config()
    rl_config.set_parameters(
            alpha=extra['alpha'],
            gamma=extra['gamma'],
            blocksize=extra['blocksize'])
    rl_config.paths_to_SARSA = getattr(sarsa_util, extra['state_functions'][0])
    rl_config.make_path_NN = getattr(sarsa_util, extra['state_functions'][1])
    rl_config.reward_function = getattr(sarsa_util, extra['state_functions'][2])
    rl_config.transition_function = getattr(sarsa_util, extra['state_functions'][3])
    rl_config.load_pointcloud(points_file)
    rl_config.load_action_files(config_dir)
    rl_config.load_path_data(path_pat, data_ids)
    rl_config.format_grid_and_paths()
    rl_config.paths_to_SARSA(rl_config)

    Q, vals = util.do_qlearn(rl_config, num_iter, memory_size)
    display.show_value(np.log(Q*1000+1))
    display.plot_1D(vals)

    plt.show()

@argh.arg('points_file',
        help='File containing point cloud data as list of points')
@argh.arg('path_pat',
        help='Filename pattern for path file data (eg. data/qm_hc{0}_{1}.txt)')
@argh.arg('data_ids', help='List of data ids', nargs='+', type=int)
@argh.arg('config_dir', help='Config directory')
@argh.arg('-a', '--alpha', help='Learning rate', default=0.5)
@argh.arg('-g', '--gamma', help='Discount factor', default=0.5)
@argh.arg('-b', '--blocksize', help='Grid block size', default=0.5)
@argh.arg('-i', '--iter', help='Number of q-learning iterations', default=1000)
@argh.arg('-m', '--memory_size', help='Total memory size', default=200)
@argh.arg('-c', '--batch_size', help='Iteration sample size', default=200)
@argh.arg('-l', '--elength', help='Episode length ', default=500)
@argh.arg('-e', '--epsilon', help='epsilon greedy parameter', default=0.9)
@argh.arg('-s', '--sigma', help='Path reward sigma', default=5000)
@argh.arg('--start', default=0, help='Z level to begin plot at')
@argh.arg('--max_div', default=8, help='Divide max by this')
@argh.arg('--state_functions', help='Functions specification',
        default=['hc_only_make_sarsa_lists','hc_only_NN','hc_only_reward','hc_only_transition'], nargs='+', type=str)
@argh.arg('--explore_functions', help='Functions specification',
        default=['hc_only_reset','hc_only_explore_step'], nargs='+', type=str)
@argh.arg('--save', default=None, help='Save configuration and results in directory')
def explore_qlearn(points_file, path_pat, data_ids, config_dir, **extra):
    "Run basic q-learning algorithm"
    num_iter = extra['iter']
    memory_size = extra['memory_size']
    batch_size = extra['batch_size']
    episode_length = extra['elength']

    rl_config = RL_Config()
    rl_config.set_parameters(
            alpha=extra['alpha'],
            gamma=extra['gamma'],
            sigma=extra['sigma'],
            epsilon=extra['epsilon'],
            blocksize=extra['blocksize'])
    rl_config.paths_to_SARSA = getattr(sarsa_util, extra['state_functions'][0])
    rl_config.make_path_NN = getattr(sarsa_util, extra['state_functions'][1])
    rl_config.reward_function = getattr(sarsa_util, extra['state_functions'][2])
    rl_config.transition_function = getattr(sarsa_util, extra['state_functions'][3])
    rl_config.get_random_state = getattr(sarsa_util, extra['explore_functions'][0])
    rl_config.explore_step = getattr(sarsa_util, extra['explore_functions'][1])
    rl_config.set_loadfiles(
            fn_points=points_file,
            fn_config=config_dir,
            fnp_path=path_pat,
            data_ids=data_ids)

    savefolder = extra['save']
    if savefolder is not None:
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        rl_config.save(savefolder)
        summpath = os.path.join(savefolder, 'summary.txt')
        f = open(summpath, 'wb')
        summ = rl_config.get_summary()
        summ += "num_iter = {0}\t\t\t// number of training iterations\n".format(num_iter)
        summ += "batch_size = {0}\t\t\t//batch train size\n".format(batch_size)
        summ += "memory_size = {0}\t\t\t//total memory size\n".format(memory_size)
        summ += "episode_length = {0}\t\t\t//length of an episode\n".format(episode_length)
        f.write(bytes(summ, 'UTF-8'))
        f.close()

    rl_config.load_data()
    rl_config.format_grid_and_paths()
    rl_config.paths_to_SARSA(rl_config)

    (Q, vals, umap) = util.do_explore_qlearn(rl_config, num_iter=num_iter,
            rand_count=batch_size, memory=memory_size, reset_episode=episode_length)

    if savefolder is not None:
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        matpath = os.path.join(savefolder,'Q-results.mat')
        scipy.io.savemat(matpath,
            {'Q':Q, 'vals':vals, 'umap':umap, 'voxel_grid':rl_config.voxel_grid})


    """
    Q[umap == 0] = -2
    display.show_value(Q, 1)
    #display.show_value(umap, 20)
    display.plot_1D(vals)
    #display.plot_path_reward(rl_config.path_NN, rl_config.voxel_grid, extra['sigma'])
    display.show_grid(rl_config.voxel_grid, extra['start'], extra['max_div'])
    display.show_action_value(Q, 5, [0])
    display.show_action_value(Q, 6, [1])


    plt.show()
    """


@argh.arg('model', help='Folder containing the model files to load')
@argh.arg('-i', '--iter', help='Number of q-learning iterations', default=1000)
@argh.arg('-m', '--memory_size', help='Iteration sample size', default=200)
@argh.arg('-l', '--elength', help='Episode length ', default=500)
def load_qlearn(model, **extra):
    num_iter = extra['iter']
    memory_size = extra['memory_size']
    episode_length = extra['elength']

    rl_config = RL_Config.load(model)
    rl_config.load_data()
    rl_config.format_grid_and_paths()
    rl_config.paths_to_SARSA(rl_config)

    Qdict = scipy.io.loadmat(os.path.join(model,'Q-results.mat'))
    Q = Qdict['Q']
    vals = Qdict['vals']
    umap = Qdict['umap']

    Q[umap == 0] = -5
    display.show_value(Q, 1)
    display.plot_1D(vals.transpose())
    display.show_action_value(Q, 5, [0])
    display.show_action_value(Q, 6, [1])

    display.show_value(umap, 22)
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan, linewidth=120)
    argh.dispatch_commands([show_points_and_path, basic_qlearn, show_denseplot,
            explore_qlearn, load_qlearn, plot_path_rewards])
