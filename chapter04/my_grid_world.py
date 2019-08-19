import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

import pdb

matplotlib.use('Agg')

# Grid World in Example 4.1
WORLD_SIZE = 4
TA_POS = [0, 0]
TB_POS = [WORLD_SIZE-1, WORLD_SIZE-1]
DISCOUNT = 1.0

# 0:left, 1:up, 2:right, 3:down
# As offset
ACTION_NUM = 4
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROBS = [0.25] * 4

# draw image function
def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)
    

# determinstic world
def step(state, action):
    # special state A and B with wormhole
    if state == TA_POS:
        # reward = 0.0
        # new_state = TA_POS
        return 0.0, TA_POS
    elif state == TB_POS:
        # reward = 0.0
        # new_state = TB_POS
        return 0.0, TB_POS

    assert action <= 3
    new_state = (np.asarray(state) + ACTIONS[action]).tolist()
    x, y = new_state
    if (0<=x<WORLD_SIZE) and (0<=y<WORLD_SIZE):
        # normal state
        reward = -1.0
    else:
        # out of the word
        reward = -1.0
        # unchanged state
        new_state = state

    return reward, new_state

def sv_4_1():
    # estimating the state values of Fig.4.1.
    # Value prediction is used.
    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=float)
    New_state_values = state_values.copy() # not inplace
    count = 0

    while True:
        # iter until converge
        max_diff = 0
        count += 1


        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # sweep through all states
                state = [i, j]
                pre_state_value = state_values[i, j]

                # to be accumulate
                v_sum_a = 0.0
                for action in range(ACTION_NUM):
                    action_prob = ACTION_PROBS[action]
                    model_prob = 1.0 # model p(s_prime, r | s, a) = 1 in a determinstic env.
                    reward, next_state = step(state, action)
                    next_i, next_j = next_state

                    # Bellman Equation for state value (action-wise)
                    # accumulate over all actions: +=
                    v_sum_a += action_prob * model_prob * (reward + DISCOUNT * state_values[next_i, next_j])
                # update state value
                New_state_values[i, j] = v_sum_a
                
        max_diff = abs(New_state_values - state_values).max()
        state_values = New_state_values.copy()
        
        if max_diff < 1e-4:
            draw_image(np.round(state_values, decimals=2))
            plt.savefig('./images/mine/figure_4_1.png')
            plt.close()
            print('Non-Inplace Iters: {}'.format(count))
            break


def sv_4_1_inplace():
    # estimating the state values of Fig.4.1.
    # Value prediction is used. (inplace operation)
    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=float)

    count = 0
    while True:
        # iter until converge
        max_diff = 0
        count += 1

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # sweep through all states
                state = [i, j]
                pre_state_value = state_values[i, j]

                # to be accumulate
                v_sum_a = 0.0
                for action in range(ACTION_NUM):
                    action_prob = ACTION_PROBS[action]
                    model_prob = 1.0 # model p(s_prime, r | s, a) = 1 in a determinstic env.
                    reward, next_state = step(state, action)
                    next_i, next_j = next_state

                    # Bellman Equation for state value (action-wise)
                    # accumulate over all actions: +=
                    v_sum_a += action_prob * model_prob * (reward + DISCOUNT * state_values[next_i, next_j])
                # update state value
                state_values[i, j] = v_sum_a
                
                sv_diff = abs(pre_state_value - state_values[i, j])
                if sv_diff > max_diff:
                    max_diff = sv_diff
        
        if max_diff < 1e-4:
            draw_image(np.round(state_values, decimals=2))
            plt.savefig('./images/mine/figure_4_1_inplace.png')
            plt.close()
            print('Inplace Iters: {}'.format(count))
            break



if __name__ == '__main__':
    sv_4_1()
    sv_4_1_inplace()
    '''
    Non-Inplace Iters: 173
    Inplace Iters: 114 (more compute and memory efficient)
    '''
