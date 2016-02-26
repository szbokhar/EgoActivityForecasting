import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import util
import load_data

def main():
    fn_points = sys.argv[1]
    fn_path = sys.argv[2]
    fn_pathlabels = sys.argv[3]
    fn_labelnames = sys.argv[4]
    fn_rlactions = sys.argv[5]

    pts, colors = load_data.get_points_and_colors(fn_points)
    label_dict = load_data.get_labeldict(fn_labelnames)
    rl_dict = load_data.get_labeldict(fn_rlactions)

    block_size = 0.5
    training_paths = []
    training_labels = []

    print("pts min max", np.min(pts, 0), np.max(pts, 0))
    for i in range(1,2):
        print(fn_path.format(i), fn_pathlabels.format(i))
        imagenames, path = load_data.get_imagenames_and_path(fn_path.format(i))
        raw_labels = load_data.get_pathlabels(fn_pathlabels.format(i))
        print("rawpath min max", np.min(path, 0), np.max(path, 0))
        training_paths.append(path)
        training_labels.append(raw_labels)

    dplot = util.make_voxel_grid(pts, colors, block_size, paths=training_paths)
    util.show_grid(dplot)

    sars_list = None
    for i in range(len(training_paths)):
        print(fn_path.format(i), fn_pathlabels.format(i))
        selpath = training_paths[i]
        print("selpath min max", np.min(selpath, 0), np.max(selpath, 0))
        sellabels = training_labels[i]
        tmp = util.path_data_to_SARS(selpath, sellabels, label_dict, rl_dict, block_size)
        if sars_list is None:
            sars_list = tmp
        else:
            sars_list = np.concatenate((sars_list, tmp), axis=0)


    # path = util.smooth_path(path, 0.5, 0.25)
    # util.make_basic_plot(pts, colors, path, 4000)

    #Q = util.do_qlearn(sars_list, dplot.shape, 0.3, 2000, 500)
    #util.show_action_value(Q)


    plt.show()




if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan, linewidth=95)
    main()
