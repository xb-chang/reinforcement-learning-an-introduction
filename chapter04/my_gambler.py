import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pdb

matplotlib.use('Agg')

'''
This execise provides a very interesting modelling method.
It shows that modelling is very important in RL
'''

GOAL = 100

# state: captial (0-100)
STATES = range(GOAL+1)

DISCOUNT = 1.0

# win by chance
# win_prob = 0.25
# win_prob = 0.4
win_prob = 0.49

# win by estimate
# win_prob = 0.5
# win_prob = 0.55

def possible_stakes(state):
    return list(range(min(state, 100-state)+1))

def figure_4_3_val_iter():
    value = np.zeros(GOAL + 1)
    value[GOAL] = 1.0 # v(0) = 0.0; v(100) = 1.0;

    policy = np.zeros(value.shape , dtype=np.int)

    sweeps_history = []
    iterations = 0
    while True:
        # value iteration
        max_diff = 0
        old_state_value = value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            old_value = value[state]
            stakes = possible_stakes(state)
            action_values = []
            for stake in stakes:
                # win
                w_state = state + stake
                assert 0 < w_state <= GOAL
                # lose
                l_state = state - stake
                assert 0 <= l_state < GOAL

                # reward = 0.0
                # mistake I made (CXB)
                # if w_state == GOAL:
                    # reward = 1.0

                tmp_value = win_prob * DISCOUNT * value[w_state] + (1.0-win_prob) * DISCOUNT * value[l_state]

                action_values.append(tmp_value)
            # pdb.set_trace()
            new_value = np.max(action_values)
            diff = abs(old_value - new_value)
            value[state] = new_value

            if max_diff < diff:
                max_diff = diff
        iterations += 1
        print('Iter {}: max diff: {:.6f}'.format(iterations, max_diff))
        if max_diff < 1e-9:
            sweeps_history.append(value)
            break

    for state in STATES[1:GOAL]:
        stakes = possible_stakes(state)
        action_values = []
        for stake in stakes:
            # win
            w_state = state + stake
            assert 0 < w_state <= GOAL
            # lose
            l_state = state - stake
            assert 0 <= l_state < GOAL

            # reward = 0.0
            # mistake I made (CXB)
            # if w_state == GOAL:
                # reward = 1.0

            tmp_value = win_prob * DISCOUNT * value[w_state] + (1.0-win_prob) * DISCOUNT * value[l_state]
            
            action_values.append(tmp_value)
        
        new_action = stakes[np.argmax(action_values)]
        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        policy[state] = stakes[np.argmax(np.round(action_values[1:], 5)) + 1]

    # pdb.set_trace()

    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('./images/mine/figure_4_3_p{}.png'.format(win_prob))
    plt.close()


if __name__ == '__main__':
    figure_4_3_val_iter()
