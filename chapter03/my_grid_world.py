import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

import pdb

matplotlib.use('Agg')

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

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
    if state == A_POS:
        # reward = 10.0
        # new_state = A_PRIME_POS
        return 10.0, A_PRIME_POS
    elif state == B_POS:
        # reward = 5.0
        # new_state = B_PRIME_POS
        return 5.0, B_PRIME_POS

    assert action <= 3
    new_state = (np.asarray(state) + ACTIONS[action]).tolist()
    x, y = new_state
    if (0<=x<WORLD_SIZE) and (0<=y<WORLD_SIZE):
        # normal state
        reward = 0.0
    else:
        # out of the word
        reward = -1.0
        # unchanged state
        new_state = state

    return reward, new_state

def sv_3_2():
    # estimating the state values of Fig.3.2.
    # Value prediction is used (Beyond Chapter 3, in Chapter 4 P75 instead)
    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=float)

    while True:
        # iter until converge
        max_diff = 0

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
            plt.savefig('./images/mine/figure_3_2.png')
            plt.close()
            break


def opt_sv_3_5():
    # Eq (3.19) / (3.20) provides the iterative method to compute optimal state/action-values directly (sweep through + max operations)
    # Indirectly have the optimal policy?
    # Different from the Generalised Policy Iterations (GPI) in Chapter 4, back and forth Prediction <-> Improvement.

    # Very similar to 3_2
    opt_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=float)
    iter = 0

    while True:
        max_diff = 0
        iter += 1

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                state = [i, j]
                pre_opt_sv = opt_state_values[i, j]
                sv_list = []

                for action in range(ACTION_NUM):
                    reward, next_state = step(state, action)
                    next_i, next_j = next_state

                    # Eq (3.19)
                    sv_list.append(reward + DISCOUNT * opt_state_values[next_i, next_j])
                # max_a rather than expectation
                assert len(sv_list) == ACTION_NUM
                opt_state_values[i, j] = np.max(sv_list)

                sv_diff = abs(pre_opt_sv - opt_state_values[i, j])
                if sv_diff > max_diff:
                    max_diff = sv_diff
            
        if max_diff < 1e-4:
            print(iter)
            print(max_diff)
            draw_image(np.round(opt_state_values, decimals=2))
            plt.savefig('./images/mine/figure_3_5.png')
            plt.close()
            break



if __name__ == '__main__':
    sv_3_2()
    opt_sv_3_5()
