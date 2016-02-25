import sys
import util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    fn_points = sys.argv[1]
    fn_path = sys.argv[2]
    fn_pathlabels = sys.argv[3]
    fn_labelnames = sys.argv[4]
    fn_rlactions = sys.argv[5]

    pts, colors = util.get_points_and_colors(fn_points)
    label_dict = util.get_labeldict(fn_labelnames)
    rl_dict = util.get_labeldict(fn_rlactions)

    block_size = 0.5
    sars_list = None
    for i in range(1,3):
        print(fn_path.format(i), fn_pathlabels.format(i))
        imagenames, path = util.get_imagenames_and_path(fn_path.format(i))
        raw_labels = util.get_pathlabels(fn_pathlabels.format(i))
        tmp = util.path_data_to_SARS(path, raw_labels, label_dict, rl_dict, block_size)
        if sars_list is None:
            sars_list = tmp
        else:
            sars_list = np.concatenate((sars_list, tmp), axis=0)

    """
    pts, dx = util.center_points(pts)
    path = path - dx
    """

    Q = util.do_qlearn(sars_list, 0.3, 2000, 500)

    util.show_action_value(Q)

    # util.make_basic_plot(pts, colors, path[0:34,:]*block_size, 8000)

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan, linewidth=95)
    main()
