import numpy as np
import scipy.spatial

def full_bagofactions(rl_config):
    print('-8-')
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

    lnames = rl_config.seq_actions
    rl_actions = rl_config.rl_actions
    final_reward = rl_config.final_reward
    med_reward = rl_config.path_reward

    ka2rl = {3:7, 4:9, 5:8, 6:10, 7:11}
    rl2id = lambda x: x-7

    for p in range(len(rl_config.paths)):
        path = rl_config.paths[p]
        sa_list = []
        boa = [0, 0, 0, 0, 0]
        nboa = [0, 0, 0, 0, 0]

        for i in range(len(path.points)-1):
            pos = path.points[i,:]
            npos = path.points[i+1,:]
            lbl_num = path.seq_labels[i]
            lbl = lnames[lbl_num]

            act = -1

            if lbl == 'Standing':
                act = 0     # Nothing
            elif lbl == 'Walking':
                act = get_direction(npos - pos)
            elif lbl_num >= 3:
                nlbl_num = path.seq_labels[i+1]
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
            sars_list[i, :] = sa_list[i][0] + [sa_list[i][1]] + [med_reward] + sa_list[i+1][0]

        sars_list[-1, 9] = final_reward

        rl_config.paths[p].SARSA_list = sars_list

    print('-0-', len(rl_config.paths))

def full_path_NN(rl_config):
    sars_list = rl_config.total_SARSA_list
    point_sets = {}

    for i in range(sars_list.shape[0]):
        S = sars_list[i]
        idx = tuple(S[3:8])
        if idx not in point_sets:
            point_sets[idx] = S[[0,2]].reshape((1,2))
        else:
            point_sets[idx] = np.concatenate((point_sets[idx], S[[0,2]].reshape((1,2))), axis=0)


    for k in point_sets:
        point_sets[k] = scipy.spatial.KDTree(point_sets[k])


    return point_sets
