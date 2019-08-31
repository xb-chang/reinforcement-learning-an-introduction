import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from random import randint

import pdb

DISCOUNT = 1.0

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

def bj_eposide_gen(p_policy, d_policy):
    '''
    p_policy: player policy, 31 length
    d_policy: dealer policy, 31 length
    '''
    # player init
    player_cards = [card_dealt(), card_dealt()]
    # dealer init
    dealer_cards = [card_dealt(), card_dealt()]

    t = 0
    dealer_show = dealer_cards[0]
    ace_heat, p_sum = sum_cards(player_cards)

    # state and action belong to t
    state_list = []
    state_list.append((p_sum, dealer_show, ace_heat))

    action_list = []
    action = p_policy[p_sum]
    action_list.append(action)
    
    # reward belong to t+1
    reward_list = [] # R_{t+1}

    
        
    if p_sum == 21 and t == 0:
        # natural: player 21 at the very begining
        # dealer sum
        _, d_sum = sum_cards(dealer_cards)
        # cmp and reward
        reward_list.append(cmp_r(p_sum, d_sum))
    else:
        # normal game
        # player's turn first
        while action == 1:
            # hit, next time
            t += 1
            # hit with reward 0
            reward_list.append(0.0)

            # new state and action
            player_cards.append(card_dealt())
            ace_heat, p_sum = sum_cards(player_cards)
            state_list.append((p_sum, dealer_show, ace_heat))

            action = p_policy[p_sum]
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




def figure_5_1():
    # simple policy with a threshold
    # This simple policy only cares the sum in hand.
    # player stick_thres = 20
    player_simple_policy = ([None] + [1] * 19 + [0] * 11) # None is occupier, bust to 31
    # dealer stick_thres = 17
    dealer_simple_policy = ([None] + [1] * 16 + [0] * 14) # None is occupier, bust to 31

    # state vaiues
    # usable ace state
    U_ACE_V = np.zeros((10, 10), dtype=np.float)
    U_ACE_C = np.zeros((10, 10), dtype=np.float)
    # no usable ace state
    N_ACE_V = np.zeros((10, 10), dtype=np.float)
    N_ACE_C = np.zeros((10, 10), dtype=np.float)
    # 200 states in total (no/usable ACE)
    # V[0]: No Usable Ace
    # V[1]: Usable Ace
    V = [N_ACE_V, U_ACE_V]
    # counting
    C = [N_ACE_C, U_ACE_C]
    
    show_states = [None,
              None,
              None,
              None]

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']

    eposide_c = 0
    while True:
        # state: (p_sum, d_show, u_ace: 0-no; 1-yes), p_sum can < 12 or bust
        # action: stick-0; hit-1
        # reward: -1, 0, 1
        # random starting point
        states, actions, rewards = bj_eposide_gen(player_simple_policy, dealer_simple_policy)
        
        # state & action correspond to t
        # reward corresponds to t+1        

        assert len(states) == len(actions) == len(rewards)
        t_len = len(states)
        
        # G_T = 0
        Return = 0.0

        for t_step in reversed(range(t_len)):
            state = states[t_step]
            p_sum, dealer_show, ace_heat = state
            reward = rewards[t_step]

            # G_t = R_{t+1} + discount * G_{t+1}
            Return = reward + DISCOUNT * Return

            # One advantage of MC method !!!!!!
            # update the meaningful states only
            if 12 <= p_sum <= 21:
                if state in states[:t_step]:
                    # not first visit here
                    pass

                else:
                    # first visit here
                    # update the very first state only 
                    assert ace_heat in [0, 1]
                    avg_v = V[ace_heat][p_sum-12, dealer_show-1]
                    count = C[ace_heat][p_sum-12, dealer_show-1]

                    V[ace_heat][p_sum-12, dealer_show-1] = (avg_v * count + Return)/(count+1.0)
                    C[ace_heat][p_sum-12, dealer_show-1] = count+1.0

        eposide_c += 1    

        if eposide_c == 10000:
            print('{} eposide Done.'.format(eposide_c))
            show_states[0] = np.copy(V[1])
            show_states[2] = np.copy(V[0])
            # pdb.set_trace()
        elif eposide_c == 500000:
            print('{} eposide Done.'.format(eposide_c))
            show_states[1] = np.copy(V[1])
            show_states[3] = np.copy(V[0])
            # pdb.set_trace()
            break

    # drawing
    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for show_state, title, axis in zip(show_states, titles, axes):
        fig = sns.heatmap(np.flipud(show_state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('./images/mine/figure_5_1.png')
    plt.close()


if __name__ == '__main__':
    figure_5_1()
    # figure_5_2()
    # figure_5_3()