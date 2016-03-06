import numpy as np
import scipy.spatial

def full_bagofactions(rl_config):
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

def hc_only(rl_config):
    sid2seq_actions = rl_config.seq_actions
    seq_actions2sid = {v: k for k, v in sid2seq_actions.items()}

    rid2rl_actions = rl_config.rl_actions
    rl_actions2rid = {v: k for k, v in rid2rl_actions.items()}

    id2rl_state = rl_config.rl_state_ids
    rl_state2id = {v: k for k, v in id2rl_state.items()}
    state_size = len(id2rl_state.keys())

    final_reward = rl_config.final_reward
    med_reward = rl_config.path_reward


    def get_direction(dx):
        x = dx[0]
        z = dx[1]
        y = dx[2]

        if x == 0 and y == 0 and z == 0:
            return rl_actions2rid['Nothing']
        elif x > 0:
            return rl_actions2rid['Move_East']
        elif x < 0:
            return rl_actions2rid['Move_West']
        elif y > 0:
            return rl_actions2rid['Move_South']
        elif y < 0:
            return rl_actions2rid['Move_North']
        elif z > 0:
            return rl_actions2rid['Move_Down']
        elif z < 0:
            return rl_actions2rid['Move_Up']


    for p in range(len(rl_config.paths)):
        path = rl_config.paths[p]
        sa_list = []

        state = np.zeros(state_size)
        next_state = np.zeros(state_size)

        for i in range(len(path.points)-1):
            pos = path.points[i,:]
            npos = path.points[i+1,:]
            pos_sid = path.seq_labels[i]

            act = -1
            state[rl_state2id['Pos_X']] = pos[0]
            state[rl_state2id['Pos_Y']] = pos[2]

            if sid2seq_actions[pos_sid] == 'Standing':
                act = rl_actions2rid['Nothing']
            elif sid2seq_actions[pos_sid] == 'Walking':
                act = get_direction(npos - pos)
            elif sid2seq_actions[pos_sid] == 'Make_Hot_Chocolate':
                if path.seq_labels[i+1] != pos_sid:
                    act = rl_actions2rid['Do_MakeHotChocolate']
                    next_state[rl_state2id['MakeHotChocolate']] += 1
                else:
                    act = get_direction(npos - pos)
            elif sid2seq_actions[pos_sid] == 'Finish':
                act = rl_actions2rid['Finish']
                next_state[rl_state2id['MakeHotChocolate']] += 1
            else:
                act = get_direction(npos - pos)

            sa_list.append((state, int(act)))
            state = np.copy(next_state)


        sars_list = np.zeros((len(sa_list)-1, 2*state_size + 2))

        for i in range(len(sa_list)-1):
            sars_list[i, :] = np.concatenate((sa_list[i][0],[sa_list[i][1]],[med_reward],sa_list[i+1][0]))

        sars_list[-1, state_size+1] = final_reward

        rl_config.paths[p].SARSA_list = sars_list

    rl_config.q_shape = [rl_config.voxel_grid.shape[0], rl_config.voxel_grid.shape[2], 3, len(rid2rl_actions.keys())]

def hc_only_NN(rl_config):
    sars_list = rl_config.total_SARSA_list
    id2rl_state = rl_config.rl_state_ids
    rl2id = {v: k for k, v in id2rl_state.items()}
    idxidx = [rl2id['MakeHotChocolate']]
    posidx = [rl2id['Pos_X'], rl2id['Pos_Y']]
    point_sets = {}

    for i in range(sars_list.shape[0]):
        S = sars_list[i]
        idx = tuple(idxidx)
        if idx not in point_sets:
            point_sets[idx] = S[posidx].reshape((1,len(posidx)))
        else:
            point_sets[idx] = np.concatenate((point_sets[idx], S[posidx].reshape((1,len(posidx)))), axis=0)

    for k in point_sets:
        point_sets[k] = scipy.spatial.KDTree(point_sets[k])


    return point_sets
