import numpy as np
import pickle as pkl

import util
import load_data

class RL_Config:
    DEF_ALPHA = 0.9
    DEF_GAMMA = 0.8
    DEF_EPSILON = 0.6
    DEF_SIGMA = 10
    DEF_BLOCKSIZE = 0.5
    DEF_SMOOTH = [0.2, 0.1]

    def __init__(self):
        self.fn_points = None
        self.fn_config = None
        self.fnp_path = None
        self.data_ids = None

        self.pc_points = None
        self.pc_colors = None

        self.seq_actions = None
        self.rl_actions = None
        self.rl_state_ids = None

        self.paths = None
        self.voxel_grid = None
        self.total_SARSA_list = None
        self.path_NN = None

        self.paths_to_SARSA = None
        self.make_path_NN = None
        self.reward_function = None
        self.get_random_state = None
        self.explore_step = None

        self.q_shape = None
        self.set_parameters()

    def get_summary(self):
        summ_text = ""
        summ_text += "alpha = {0}\t\t\t//Learning rate\n".format(self.alpha)
        summ_text += "gamma = {0}\t\t\t//Propogation constants\n".format(self.gamma)
        summ_text += "epsilon = {0}\t\t\t//e-greedy constant\n".format(self.epsilon)
        summ_text += "sigma = {0}\t\t\t//Path reward spread\n".format(self.sigma)
        summ_text += "blocksize = {0}\t\t\t//side length of one grid block\n".format(self.blocksize)
        summ_text += "smooth = {0}\t\t\t//path smoothing alpha and beta values\n".format(self.smooth)
        return summ_text

    def save(self, fname):
        pkl.dump(self, open(fname+'config.pkl', 'wb'))

    @staticmethod
    def load(fname):
        return pkl.load(open(fname+'config.pkl', 'rb'))

    def set_loadfiles(self, fn_points=None, fn_config=None, fnp_path=None, data_ids=None):
        self.fn_points = fn_points
        self.fn_config = fn_config
        self.fnp_path = fnp_path
        self.data_ids = data_ids

    def load_data(self):
        self.load_pointcloud(self.fn_points)
        self.load_action_files(self.fn_config)
        self.load_path_data(self.fnp_path, self.data_ids)

    def load_action_files(self, config_dir):
        self.seq_actions = load_data.get_labeldict(config_dir+'/sequence_actions.txt')
        self.rl_actions = load_data.get_labeldict(config_dir+'/rl_actions.txt')
        self.rl_state_ids = load_data.get_labeldict(config_dir+'/rl_state_ids.txt')

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

    def make_total_SARSA_list(self):
        self.total_SARSA_list = None

        for i in range(len(self.paths)):
            if self.total_SARSA_list is None:
                self.total_SARSA_list = np.copy(self.paths[i].SARSA_list)
            else:
                self.total_SARSA_list = np.concatenate((self.total_SARSA_list, self.paths[i].SARSA_list), axis=0)

    def set_parameters(self, alpha=DEF_ALPHA, gamma=DEF_GAMMA, epsilon=DEF_EPSILON,
            sigma=DEF_SIGMA, blocksize=DEF_BLOCKSIZE, smooth_params=DEF_SMOOTH):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma = sigma
        self.blocksize = blocksize
        self.smooth = smooth_params

class Path:
    def __init__(self, points, images, labels):
        self.points = points
        self.raw_points = points
        self.smooth_points = points
        self.block_points = points
        self.imagenames = images
        self.seq_labels = labels
        self.SARSA_list = None
