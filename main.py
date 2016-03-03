import sys
import argh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import util
import load_data
import display

@argh.arg('points_file', help='File containing point cloud data as list of points')
@argh.arg('-b', '--blocksize', default=0.5, help='Side length of grid cube')
@argh.arg('-s', '--start', default=0, help='Z level to begin plot at')
@argh.arg('-m', '--max_div', default=8, help='Divide max by this')
def show_denseplot(points_file, **extra):
    "Generate and show pointclound density plot"
    pts, colors = load_data.get_points_and_colors(points_file)

    block_size = extra['blocksize']

    dplot = util.make_voxel_grid(pts, colors, block_size)
    display.show_grid(dplot, extra['start'], extra['max_div'])

    plt.show()

@argh.arg('points_file', help='File containing point cloud data as list of points')
@argh.arg('path_file', help='File containing point path data')
@argh.arg('-c', '--count', default=4000, help='Number of points to plot')
@argh.arg('-b', '--blocksize', default=0.5, help='Side length of grid cube')
def show_points_and_path(points_file, path_file, **extra):
    """
        Loads points and path data files and plots them
    """
    pts, colors = load_data.get_points_and_colors(points_file)

    block_size = extra['blocksize']
    count = extra['count']

    imagenames, path = load_data.get_imagenames_and_path(path_file)

    path = util.smooth_path(path, 0.5, 0.25)
    bpath = util.block_path(path, block_size)
    display.make_basic_plot(pts, colors, [path, bpath*block_size], ['b-', 'r-'], count)

    plt.show()


@argh.arg('points_file', help='File containing point cloud data as list of points')
@argh.arg('path_pat', help='Filename pattern for path file data (eg. data/qm_hc{0}_{1}.txt)')
@argh.arg('data_ids', help='List of data ids', nargs='+', type=int)
@argh.arg('label_names', help='Action sequence label names file')
@argh.arg('rl_actions', help='RL actions file')
@argh.arg('-a', '--alpha', help='Learning rate', default=0.5)
@argh.arg('-g', '--gamma', help='Discount factor', default=0.5)
@argh.arg('-b', '--block_size', help='Grid block size', default=0.5)
@argh.arg('-i', '--iter', help='Number of q-learning iterations', default=1000)
@argh.arg('-m', '--memory_size', help='Iteration sample size', default=200)
@argh.arg('-r', '--rewards', help='Final and intermediate path rewards', default=[1,0], nargs='+', type=int)
def basic_qlearn(points_file, path_pat, data_ids, label_names, rl_actions, **extra):
    "Run basic q-learning algorithm"
    fn_points = points_file
    fn_path = path_pat

    pts, colors = load_data.get_points_and_colors(fn_points)
    label_dict = load_data.get_labeldict(label_names)
    rl_dict = load_data.get_labeldict(rl_actions)

    print(extra)

    block_size = extra['block_size']
    alpha = extra['alpha']
    gamma = extra['gamma']
    num_iter = extra['iter']
    rewards = extra['rewards']
    memory_size = extra['memory_size']
    training_paths = []
    training_labels = []

    for i in range(len(data_ids)):
        pathfile = fn_path.format(data_ids[i], 'path')
        labelsfile =  fn_path.format(data_ids[i], 'labels')
        print(pathfile, labelsfile)
        imagenames, path = load_data.get_imagenames_and_path(pathfile)
        raw_labels = load_data.get_pathlabels(labelsfile)
        training_paths.append(path)
        training_labels.append(raw_labels)

    dplot = util.make_voxel_grid(pts, colors, block_size, paths=training_paths)

    sars_list = None
    for i in range(len(training_paths)):
        selpath = training_paths[i]
        sellabels = training_labels[i]
        tmp, _ = util.path_data_to_SARS(selpath, sellabels, label_dict, rl_dict, block_size, rewards[0], rewards[1])
        if sars_list is None:
            sars_list = tmp
        else:
            sars_list = np.concatenate((sars_list, tmp), axis=0)


    Q = util.do_qlearn(sars_list, dplot.shape, alpha, num_iter, memory_size, gamma)
    display.show_value(Q)

    plt.show()

@argh.arg('points_file', help='File containing point cloud data as list of points')
@argh.arg('path_pat', help='Filename pattern for path file data (eg. data/qm_hc{0}_{1}.txt)')
@argh.arg('data_ids', help='List of data ids', nargs='+', type=int)
@argh.arg('label_names', help='Action sequence label names file')
@argh.arg('rl_actions', help='RL actions file')
@argh.arg('-a', '--alpha', help='Learning rate', default=0.5)
@argh.arg('-g', '--gamma', help='Discount factor', default=0.5)
@argh.arg('-b', '--block_size', help='Grid block size', default=0.5)
@argh.arg('-i', '--iter', help='Number of q-learning iterations', default=1000)
@argh.arg('-m', '--memory_size', help='Iteration sample size', default=200)
@argh.arg('-r', '--rewards', help='Final and intermediate path rewards', default=[1,0], nargs='+', type=int)
@argh.arg('-l', '--elength', help='Episode length ', default=500)
@argh.arg('-e', '--epsilon', help='epsilon greedy parameter', default=0.9)
@argh.arg('-s', '--sigma', help='Path reward sigma', default=5000)
def explore_qlearn(points_file, path_pat, data_ids, label_names, rl_actions, **extra):
    "Run basic q-learning algorithm"
    fn_points = points_file
    fn_path = path_pat

    pts, colors = load_data.get_points_and_colors(fn_points)
    label_dict = load_data.get_labeldict(label_names)
    rl_dict = load_data.get_labeldict(rl_actions)

    block_size = extra['block_size']
    alpha = extra['alpha']
    gamma = extra['gamma']
    sigma = extra['sigma']
    num_iter = extra['iter']
    memory_size = extra['memory_size']
    rewards = extra['rewards']
    training_paths = []
    training_labels = []

    for i in range(len(data_ids)):
        pathfile = fn_path.format(data_ids[i], 'path')
        labelsfile =  fn_path.format(data_ids[i], 'labels')
        print(pathfile, labelsfile)
        imagenames, path = load_data.get_imagenames_and_path(pathfile)
        raw_labels = load_data.get_pathlabels(labelsfile)
        training_paths.append(path)
        training_labels.append(raw_labels)

    dplot = util.make_voxel_grid(pts, colors, block_size, paths=training_paths)

    sars_list = None
    final_states = None
    for i in range(len(training_paths)):
        selpath = training_paths[i]
        sellabels = training_labels[i]
        tmp, final = util.path_data_to_SARS(selpath, sellabels, label_dict, rl_dict,
                block_size, rewards[0], rewards[1])
        if final_states is None:
            final_states = final.reshape((1, final.shape[0]))
        else:
            final_states = np.concatenate((final_states, final.reshape((1,final.shape[0]))),0)

        if sars_list is None:
            sars_list = tmp
        else:
            sars_list = np.concatenate((sars_list, tmp), axis=0)

    point_sets = util.get_paths_tree(sars_list)

    (Q, umap) = util.do_explore_qlearn(sars_list, final_states, dplot, point_sets, rl_dict,
            alpha=alpha, num_iter=num_iter, rand_count=memory_size,
            reset_episode=extra['elength'], epsilon=extra['epsilon'], gamma=gamma, sigma=sigma)
    display.show_value(Q)
    display.show_action_value(Q, 4, [0,0,0,0,0])
    display.show_action_value(Q, 5, [1,0,0,0,0])
    display.show_action_value(Q, 6, [1,0,1,0,0])
    display.show_action_value(Q, 7, [1,1,1,0,0])
    display.show_action_value(Q, 8, [1,1,1,1,0])
    display.show_action_value(Q, 9, [2,1,1,1,0])
    display.show_action_value(Q, 10, [2,1,1,1,1])
    display.show_action_value(Q, 11, [2,2,1,1,1])
    display.plot_path_reward(point_sets, dplot, sigma=sigma)

    fig = plt.figure(3)
    a = fig.add_subplot(1,1,1)
    umap = np.log(umap+1)
    plt.imshow(umap, cmap=plt.get_cmap('inferno'), vmin=np.min(umap), vmax=np.max(umap))

    plt.show()

@argh.arg('points_file', help='File containing point cloud data as list of points')
@argh.arg('path_pat', help='Filename pattern for path file data (eg. data/qm_hc{0}_{1}.txt)')
@argh.arg('data_ids', help='List of data ids', nargs='+', type=int)
@argh.arg('label_names', help='Action sequence label names file')
@argh.arg('rl_actions', help='RL actions file')
@argh.arg('-b', '--block_size', help='Grid block size', default=0.5)
@argh.arg('-r', '--rewards', help='Final and intermediate path rewards', default=[1,0], nargs='+', type=int)
@argh.arg('-s', '--sigma', help='Path reward sigma', default=5000)
def plot_path_rewards(points_file, path_pat, data_ids, label_names, rl_actions, **extra):
    "Run basic q-learning algorithm"
    fn_points = points_file
    fn_path = path_pat

    pts, colors = load_data.get_points_and_colors(fn_points)
    label_dict = load_data.get_labeldict(label_names)
    rl_dict = load_data.get_labeldict(rl_actions)

    block_size = extra['block_size']
    rewards = extra['rewards']
    training_paths = []
    training_labels = []

    for i in range(len(data_ids)):
        pathfile = fn_path.format(data_ids[i], 'path')
        labelsfile =  fn_path.format(data_ids[i], 'labels')
        print(pathfile)
        imagenames, path = load_data.get_imagenames_and_path(pathfile)
        raw_labels = load_data.get_pathlabels(labelsfile)
        training_paths.append(path)
        training_labels.append(raw_labels)

    dplot = util.make_voxel_grid(pts, colors, block_size, paths=training_paths)

    sars_list = None
    for i in range(len(training_paths)):
        selpath = training_paths[i]
        sellabels = training_labels[i]
        tmp, final = util.path_data_to_SARS(selpath, sellabels, label_dict, rl_dict,
                block_size, rewards[0], rewards[1])

        if sars_list is None:
            sars_list = tmp
        else:
            sars_list = np.concatenate((sars_list, tmp), axis=0)

    point_sets = util.get_paths_tree(sars_list)

    display.plot_path_reward(point_sets, dplot, extra['sigma'])

    plt.show()

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan, linewidth=120)
    argh.dispatch_commands([show_points_and_path, basic_qlearn, show_denseplot,
            explore_qlearn, plot_path_rewards])
