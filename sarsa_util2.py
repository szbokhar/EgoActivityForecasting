import numpy as np
import scipy.spatial
from numpy.random import rand, choice, randint

def hc_only_make_sarsa_lists(rl_config):
    sid2seq_actions = rl_config.seq_actions
    seq_actions2sid = {v: k for k, v in sid2seq_actions.items()}

    rid2rl_actions = rl_config.rl_actions
    rl_actions2rid = {v: k for k, v in rid2rl_actions.items()}

    id2rl_state = rl_config.rl_state_ids
    rl_state2id = {v: k for k, v in id2rl_state.items()}
    state_size = len(id2rl_state.keys())

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
        state[rl_state2id['BeforeHC']] = 1
        next_state = np.zeros(state_size)
        next_state[rl_state2id['BeforeHC']] = 1

        for i in range(len(path.points)-1):
            pos = path.points[i,:]
            npos = path.points[i+1,:]
            pos_sid = path.seq_labels[i]

            act = -1
            state[rl_state2id['Pos_X']] = pos[0]
            state[rl_state2id['Pos_Y']] = pos[2]

            if sid2seq_actions[pos_sid] == 'Standing':
                act = get_direction(npos-pos)
            elif sid2seq_actions[pos_sid] == 'Walking':
                act = get_direction(npos - pos)
            elif sid2seq_actions[pos_sid] == 'Make_Hot_Chocolate':
                if path.seq_labels[i+1] != pos_sid:
                    act = rl_actions2rid['Do_MakeHotChocolate']
                    next_state[rl_state2id['AfterHC']] = 1
                    next_state[rl_state2id['BeforeHC']] = 0
                else:
                    act = get_direction(npos - pos)
            elif sid2seq_actions[pos_sid] == 'Finish':
                act = rl_actions2rid['Finish']
                next_state[rl_state2id['AfterHC']] = 0
                next_state[rl_state2id['Finished']] = 1
            else:
                act = get_direction(npos - pos)

            sa_list.append((state, int(act)))
            state = np.copy(next_state)


        sars_list = np.zeros((0, 2*state_size + 2))

        for i in range(len(sa_list)-1):
            state = sa_list[i][0]
            action = sa_list[i][1]
            new_state = sa_list[i+1][0]
            if action != rl_actions2rid['Nothing']:
                sars_list = np.concatenate((sars_list,
                        np.concatenate((sa_list[i][0],[sa_list[i][1]],[0],sa_list[i+1][0])).reshape((1, 2*state_size+2))))

        rl_config.paths[p].SARSA_list = sars_list

    rl_config.make_total_SARSA_list()

    rl_config.path_NN = rl_config.make_path_NN(rl_config)

    for p in range(len(rl_config.paths)):
        sars_list = rl_config.paths[p].SARSA_list

        for i in range(len(sars_list)):
            state = sars_list[i, 0:state_size]
            action = sars_list[i, state_size]
            new_state = sars_list[i, (state_size+2):(2*state_size+2)]
            sars_list[i, state_size+1] = rl_config.reward_function(rl_config, state, action, new_state)

    rl_config.make_total_SARSA_list()

    print(rl_config.total_SARSA_list)


    rl_config.q_shape = [rl_config.voxel_grid.shape[0], rl_config.voxel_grid.shape[2], 3, len(rid2rl_actions.keys())]

def hc_only_NN(rl_config):
    sars_list = rl_config.total_SARSA_list
    id2rl_state = rl_config.rl_state_ids
    rl2id = {v: k for k, v in id2rl_state.items()}
    idxidx = [rl2id['BeforeHC'], rl2id['AfterHC'], rl2id['Finished']]
    posidx = [rl2id['Pos_X'], rl2id['Pos_Y']]
    point_sets = {}

    for i in range(sars_list.shape[0]):
        S = sars_list[i]
        idx = list(idxidx)
        if tuple(S[idx]) not in point_sets:
            point_sets[tuple(S[idx])] = S[posidx].reshape((1,len(posidx)))
        else:
            point_sets[tuple(S[idx])] = np.concatenate((point_sets[tuple(S[idx])], S[posidx].reshape((1,len(posidx)))), axis=0)

    for k in point_sets:
        point_sets[k] = scipy.spatial.KDTree(point_sets[k])

    return point_sets

def hc_only_reset(rl_config):
    id2rl_state = rl_config.rl_state_ids
    rl_state2id = {v: k for k, v in id2rl_state.items()}
    state_size = len(rl_config.rl_state_ids.keys())

    e_state = np.zeros(state_size)
    e_state[rl_state2id['Pos_X']] = randint(rl_config.voxel_grid.shape[0])
    e_state[rl_state2id['Pos_Y']] = randint(rl_config.voxel_grid.shape[2])
    e_state[rl_state2id['MakeHotChocolate']] = randint(2)

    return e_state

def hc_only_explore_step(rl_config, Q, state, epsilon=0.9):
    rid2rl_actions = rl_config.rl_actions
    id2rl_state = rl_config.rl_state_ids
    rl_state2id = {v: k for k, v in id2rl_state.items()}

    act = -1
    while (act == -1):
        is_greed = rand(1) < epsilon
        if is_greed:
            idx = state.tolist() + [[x for x in range(len(rid2rl_actions.keys()))]]
            act = np.argmax(Q[tuple(idx)])
            val = np.max(Q[tuple(idx)])
            if val <= 0.00000000001:
                is_greed = False

        if not is_greed:
            if rand(1) < 0.9:
                act = randint(0, 7)
            else:
                act = randint(7, len(rid2rl_actions.keys()))

        (next_state, act, isFinished) = rl_config.transition_function(rl_config, state, act, Q)

    reward = rl_config.reward_function(rl_config, state, act, next_state)
    sarsa_state = np.concatenate((state, [act], [reward], next_state))
    length = sarsa_state.shape[0]
    sarsa_state = np.reshape(sarsa_state, (1,length))

    return (next_state, sarsa_state, isFinished)

def hc_only_reward(rl_config, state, action, new_state):
    state_size = len(rl_config.rl_state_ids.keys())
    rid2rl_actions = rl_config.rl_actions
    rl_actions2rid = {v: k for k, v in rid2rl_actions.items()}
    id2rl_state = rl_config.rl_state_ids
    rl2id = {v: k for k, v in id2rl_state.items()}

    sarsa = rl_config.total_SARSA_list
    if rl_config.hc_pos is None:
        hc_data = np.zeros((0,2))
        for i in range(sarsa.shape[0]):
            if rid2rl_actions[sarsa[i,state_size]] == "Do_MakeHotChocolate":
                hc_data = np.concatenate((hc_data, sarsa[i, [rl2id['Pos_X'],rl2id['Pos_Y']]].reshape((1,2))), axis=0)

        rl_config.hc_pos = scipy.spatial.KDTree(hc_data)


    hcpos = rl_config.hc_pos

    idxidx = [rl2id['BeforeHC'], rl2id['AfterHC'], rl2id['Finished']]
    posidx = [rl2id['Pos_X'], rl2id['Pos_Y']]
    x = state[rl2id['Pos_X']]
    y = state[rl2id['Pos_Y']]

    goalR = rl_config.rewards['Goal']
    actionP = rl_config.rewards['ActionPenalty']
    wallP = rl_config.rewards['WallPenalty']

    reward = 0
    dist,_ = rl_config.path_NN[tuple(state[list(idxidx)])].query(state[posidx])
    grid = rl_config.voxel_grid
    column = rl_config.person_column
    reward += -wallP*(np.max(grid[state[rl2id['Pos_X']],column,state[rl2id['Pos_Y']]])/np.max(grid))

    if action == rl_actions2rid['Do_MakeHotChocolate']:
        (adist,_) = hcpos.query(state[posidx])
        reward += -adist*actionP

    for p in rl_config.paths:
        print('tic')
        if np.all(new_state == p.SARSA_list[-1, (state_size+2):(2*state_size+2)]):
            print('toc')
            reward = goalR

    return reward

def hc_only_transition(rl_config, state, act, Q):
    rid2rl_actions = rl_config.rl_actions
    id2rl_state = rl_config.rl_state_ids
    rl_state2id = {v: k for k, v in id2rl_state.items()}
    maxX = Q.shape[rl_state2id['Pos_X']]
    maxY = Q.shape[rl_state2id['Pos_Y']]

    next_state = np.copy(state)
    isFinished = False
    x = next_state[rl_state2id['Pos_X']]
    y = next_state[rl_state2id['Pos_Y']]
    if rid2rl_actions[act] == 'Nothing':
        act = -1
    elif rid2rl_actions[act] == 'Move_North':
        next_state[rl_state2id['Pos_Y']] = max(y-1,0)
    elif rid2rl_actions[act] == 'Move_East':
        next_state[rl_state2id['Pos_X']] = min(x+1,maxX-1)
    elif rid2rl_actions[act] == 'Move_South':
        next_state[rl_state2id['Pos_Y']] = min(y+1,maxY-1)
    elif rid2rl_actions[act] == 'Move_West':
        next_state[rl_state2id['Pos_X']] = max(x-1,0)
    elif rid2rl_actions[act] == 'Move_Up':
        act = -1
    elif rid2rl_actions[act] == 'Move_Down':
        act = -1
    elif rid2rl_actions[act] == 'Do_MakeHotChocolate':
        next_state[rl_state2id['MakeHotChocolate']] = 1
    elif rid2rl_actions[act] == 'Finish':
        if state[rl_state2id['MakeHotChocolate']] == 1:
            next_state[rl_state2id['MakeHotChocolate']] = 2
            isFinished = True
        else:
            act = -1

    return (next_state, act, isFinished)
