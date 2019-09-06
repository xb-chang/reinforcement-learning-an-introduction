import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from random import randint

import pdb

DISCOUNT = 1.0

# Monte Carlo ES (Exploring Starts) for GPI
# The Action values Q(s, a) are estimated. Optimal equals to State values
# Exploring Starts method is used.

# A comprehensive state is (current sum, dealer's showning card, whether a usable ace hold)
# current sum < 12, always hit; activated states are [12-21]
# dealer's showing card: ACE - 10
# useable ace or not in player (keep changing every time)

def card_dealt():
    # ACE - 10
    return randint(1, 10)

def cmp_r(p_sum, d_sum):
    # compare palyer sum and dealer sum
    assert 1 <= p_sum <= 21
    assert 1 <= d_sum <= 21
    if p_sum > d_sum:
        # win
        return 1.0
    elif p_sum == d_sum:
        # drawing
        return 0.0
    else:
        # lose
        return -1.0

def sum_cards(card_list):
    # summing cards in hand
    # return: ace_heat, c_sum
    if 1 in card_list:
        no_use_ace_sum = np.asarray(card_list).sum()
        # only one ace is usable
        use_ace_sum = no_use_ace_sum + 10
        if use_ace_sum > 21:
            # cannot go bust with usable ace
            # no usable ace, 
            return 0, no_use_ace_sum
        else:
            return 1, use_ace_sum

    else:
        # no ace at all
        return 0, np.asarray(card_list).sum()

def parse_two_cards(p_sum):
    if p_sum == 21:
        ace_heat = 1
    else:
        ace_heat = randint(0, 1)

    if ace_heat == 1:
        player_cards = [1, p_sum-11]
    else:
        card1 = randint(p_sum-10, 10)
        card2 = p_sum - card1
        player_cards = [card1, card2]
    return player_cards, ace_heat


def bj_eposide_ES(p_policy, d_policy):
    '''
    black jack eposide generation with exploring start.
    p_policy: player policy, a full policy here
    d_policy: dealer policy, 31 length
    '''

    # player init
    t = 0
    # p_sum = randint(12, 21)
    # player_cards, ace_heat = parse_two_cards(p_sum)    
    player_cards = [card_dealt(), card_dealt()]
    ace_heat, p_sum = sum_cards(player_cards)

    # dealer init
    dealer_cards = [card_dealt(), card_dealt()]
    dealer_show = dealer_cards[0]
    

    # state and action belong to t
    state_list = []
    state_list.append((p_sum, dealer_show, ace_heat))

    # starting action is exploring
    action_list = []
    # action = randint(0, 1)
    # action_list.append(action)
    
    # reward belong to t+1
    reward_list = [] # R_{t+1}

    if p_sum == 21 and t == 0:
        # natural: player 21 at the very begining
        action = 0
        action_list.append(action)
        # dealer sum
        _, d_sum = sum_cards(dealer_cards)
        # cmp and reward
        reward_list.append(cmp_r(p_sum, d_sum))
    else:
        # normal game
        # player's turn first
        # ES A0
        if t == 0:
            if p_sum < 12:
                action = 1
            if p_sum > 21:
                raise ValueError('Impossoble at the beginning.')
            else:
                action = randint(0, 1)
            action_list.append(action)
        

        while action == 1:
            # hit, next time
            t += 1
            # hit with reward 0
            reward_list.append(0.0)

            # new state and action
            player_cards.append(card_dealt())
            ace_heat, p_sum = sum_cards(player_cards)
            state_list.append((p_sum, dealer_show, ace_heat))

            if p_sum < 12:
                action = 1
            elif p_sum > 21:
                action = 0
            else:
                # p_sum, dealer_show, usable_ace
                action = p_policy[p_sum-12][dealer_show-1][ace_heat]
            action_list.append(action)
        
        # action == 0, stick
        # get the final reward
        if p_sum > 21:
            # goes bust
            assert ace_heat == 0
            # loss
            reward_list.append(-1.0)
        else:
            # meet player's policy to stick
            # dealer's term
            _, d_sum = sum_cards(dealer_cards)
            d_action = d_policy[d_sum]

            while d_action == 1:
                dealer_cards.append(card_dealt())
                _, d_sum = sum_cards(dealer_cards)
                d_action = d_policy[d_sum]
            
            if d_sum > 21:
                # dealer goes bust, win
                reward_list.append(1.0)
            else:
                # compare
                reward_list.append(cmp_r(p_sum, d_sum))

    
    assert len(state_list) == len(action_list) == len(reward_list) == t+1, (len(state_list), len(action_list), len(reward_list), t)
    return state_list, action_list, reward_list




def figure_5_2():
    # A comprehensive state is (current sum, dealer's showning card, whether a usable ace hold)
    # current sum < 12, always hit; activated states are [12-21]
    # dealer's showing card: ACE - 10
    # useable ace or not in player (keep changing every time) (usable: 1; no: 0)

    # stick 0; hit: 1
    # initially, always stick
    # player_policy = np.zeros((10, 10, 2))
    player_policy = np.random.randint(2, size=(10, 10, 2))
    # dealer stick_thres = 17 (simple policy with a threshold)
    dealer_simple_policy = ([None] + [1] * 16 + [0] * 14) # None is occupier, bust to 31

    # State-Action Values
    # Q_Vs = np.zeros((10, 10, 2, 2), dtype=np.float)
    Q_Vs = np.random.rand(10, 10, 2, 2)
    C = np.zeros((10, 10, 2, 2), dtype=np.float)
    
    eposide_c = 0
    while True:
        old_Q = Q_Vs.copy()
        old_p = player_policy.copy()
        # state: (p_sum, d_show, u_ace: 0-no; 1-yes), p_sum can < 12 or bust
        # action: stick-0; hit-1
        # reward: -1, 0, 1
        # random starting point
        states, actions, rewards = bj_eposide_ES(player_policy, dealer_simple_policy)
        
        # state & action correspond to t
        # reward corresponds to t+1        

        assert len(states) == len(actions) == len(rewards)
        t_len = len(states)
        
        # G_T = 0
        Return = 0.0

        for t_step in reversed(range(t_len)):
            state = states[t_step]
            p_sum, dealer_show, ace_heat = state
            action = actions[t_step]
            reward = rewards[t_step]

            # G_t = R_{t+1} + discount * G_{t+1}
            Return = reward + DISCOUNT * Return

            # One advantage of MC method !!!!!!
            # update the meaningful states only
            # In black jack, the states are always first visit
            if 12 <= p_sum <= 21:
                if state in states[:t_step]:
                    # not first visit here
                    pass
                else:
                    # first visit here
                    # update the very first state only 
                    assert ace_heat in [0, 1]
                    assert action in [0, 1]
                    avg_v = Q_Vs[p_sum-12][dealer_show-1][ace_heat][int(action)]
                    count = C[p_sum-12][dealer_show-1][ace_heat][int(action)]

                    # policy estimation
                    Q_Vs[p_sum-12][dealer_show-1][ace_heat][int(action)] = (avg_v * count + Return)/(count+1.0)
                    C[p_sum-12][dealer_show-1][ace_heat][int(action)] = count+1.0
                    # policy improvement
                    player_policy[p_sum-12][dealer_show-1][ace_heat] = np.argmax(Q_Vs[p_sum-12][dealer_show-1][ace_heat])
        max_value_change = abs(old_Q - Q_Vs).max()
        eposide_c += 1        
        print('{} max value change {:.6f}'.format(eposide_c, max_value_change))
        # if max_value_change < 1e-4 and np.array_equal(old_p, player_policy):
        # # if max_value_change < 1e-2:
        #     break
        # previous convergence conditions failed (not stable)
        if eposide_c >= 2000000 and np.array_equal(old_p, player_policy):
            # Bellman Optimality Equation V_{*}(s) = max_{a} Q_{*}(s, a)
            state_values = np.amax(Q_Vs, axis=-1)
            opt_policy = np.argmax(Q_Vs, axis=-1)
            break
    
    action_no_usable_ace = opt_policy[:, :, 0]
    action_usable_ace = opt_policy[:, :, 1]
    state_value_no_usable_ace = state_values[:, :, 0]
    state_value_usable_ace = state_values[:, :, 1]
    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('./images/mine/figure_5_2.png')
    plt.close()


if __name__ == '__main__':
    # figure_5_1()
    figure_5_2()
    # figure_5_3()