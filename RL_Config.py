import numpy as np

import util
import load_data

class RL_Config:
    DEF_ALPHA = 0.9
    DEF_GAMMA = 0.8
    DEF_EPSILON = 0.6
    DEF_SIGMA = 10
    DEF_BLOCKSIZE = 0.5
    DEF_FINALREWARD = 100
    DEF_PATHREWARD = 10
    DEF_SMOOTH = [0.2, 0.1]

    def __init__(self):
        self.pc_points = None
        self.pc_colors = None
        self.seq_actions = None
        self.rl_actions = None
        self.paths = None
        self.voxel_grid = None
        self.total_SARS_list = None
        self.path_NN = None
        self.set_parameters()


    def load_action_files(self, fn_sequence_actions, fn_rl_actions):
        self.seq_actions = load_data.get_labeldict(fn_sequence_actions)
        self.rl_actions = load_data.get_labeldict(fn_rl_actions)

    def load_pointcloud(self, fn_pointcloud):
        pts, colors = load_data.get_points_and_colors(fn_pointcloud)
        self.pc_points = pts
        self.pc_colors = colors

    def load_path_data(self, fn_path_pattern, data_ids):
        self.paths = []
        for i in range(len(data_ids)):
            pathfile = fn_path_pattern.format(data_ids[i], 'path')
            labelsfile =  fn_path_pattern.format(data_ids[i], 'labels')
            print("Loading {0} and {1}".format(pathfile, labelsfile))

            imagenames, path = load_data.get_imagenames_and_path(pathfile)
            raw_labels = load_data.get_pathlabels(labelsfile)

            self.paths.append(Path(path, imagenames, raw_labels))

    def format_grid_and_paths(self):
        self.voxel_grid = util.make_voxel_grid(self.pc_points, self.pc_colors,
                self.blocksize, paths=self.paths, alpha=self.smooth[0], beta=self.smooth[1])

    def generate_sars_data(self):
        for i in range(len(self.paths)):
            tmp, final = util.path_data_to_SARS(self.paths[i], self.seq_actions, self.rl_actions,
                    self.blocksize, self.final_reward, self.path_reward)
            self.paths[i].SARS_list = tmp

            if self.total_SARS_list is None:
                self.total_SARS_list = np.copy(tmp)
            else:
                self.total_SARS_list = np.concatenate((self.total_SARS_list, tmp), axis=0)

        self.path_NN = util.get_paths_tree(self.total_SARS_list)


    def set_parameters(self, alpha=DEF_ALPHA, gamma=DEF_GAMMA, epsilon=DEF_EPSILON,
            sigma=DEF_SIGMA, blocksize=DEF_BLOCKSIZE, final_reward=DEF_FINALREWARD,
            path_reward=DEF_PATHREWARD, smooth_params=DEF_SMOOTH):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma = sigma
        self.blocksize = blocksize
        self.final_reward = final_reward
        self.path_reward = path_reward
        self.smooth = smooth_params

class Path:
    def __init__(self, points, images, labels):
        self.points = points
        self.raw_points = points
        self.smooth_points = points
        self.block_points = points
        self.imagenames = images
        self.seq_labels = labels
        self.SARS_list = None
