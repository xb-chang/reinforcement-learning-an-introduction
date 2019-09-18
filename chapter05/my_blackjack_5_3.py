'''
Off-policy Estimation of a Blackjack State Value
The target policy is fixed since it is a Estimation problem.
The behaviour policy is fixed as well.
'''

'''
Starting state:
dealer shows: 2;
usable ace;
player's sum is 13: [A+A+A; A+2]

t policy: stick when >= 20
b policy: random hit/stick (50/50)

GT_State_Value = -0.27726
'''

from blackjack import play, get_card, behavior_policy_player

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy

import random
from random import randint

import pdb

DISCOUNT = 1.0

def card_dealt():
    # ACE - 10, J, Q, K
    # return min(10, randint(1, 13))
    return min(10, randint(1, 14))

# # get a new card
# def get_card():
#     card = np.random.randint(1, 14)
#     card = min(card, 10)
#     return card

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

def rollDice():
    roll = random.randint(1,1000)

    if roll <= 500:
        return 0
    elif roll >= 501:
        return 1

def bj_ep_starting_state(start_states, dealer_policy):
    t = 0
    # behaviour policy is random stick/hit
    # start point choice
    # start_idx = random.choice([0,1])
    start_idx = rollDice()
    assert len(start_states) == 2
    start_state = deepcopy(start_states[start_idx])

    dealer_show = start_state[1]
    assert dealer_show == 2
    dealer_cards = [dealer_show, card_dealt()]

    player_cards = deepcopy(start_state[0])
    ace_heat, p_sum = sum_cards(player_cards)
    assert ace_heat == 1, player_cards
    assert p_sum == 13

    # state and action belong to t
    state_list = []
    state_list.append((p_sum, dealer_show, ace_heat))

    action_list = []
    # action = randint(0, 1)
    action = rollDice()
    action_list.append(action)
    
    # reward belong to t+1
    reward_list = [] # R_{t+1}

    while action == 1:
        # hit, next time
        t += 1
        # hit with reward 0
        reward_list.append(0.0)

        # new state and action
        player_cards.append(card_dealt())
        ace_heat, p_sum = sum_cards(player_cards)
        state_list.append((p_sum, dealer_show, ace_heat))

        if p_sum >= 21:
            action = 0
        else:
            action = randint(0, 1)
        action_list.append(action)
    
    # action == 0, stick
    # get the final reward
    if p_sum > 21:
        # goes bust
        assert ace_heat == 0
        # loss
        reward_list.append(-1.0)
    else:
        # player's policy to stick
        # dealer's term
        _, d_sum = sum_cards(dealer_cards)
        d_action = dealer_policy[d_sum]

        while d_action == 1:
            dealer_cards.append(card_dealt())
            _, d_sum = sum_cards(dealer_cards)
            d_action = dealer_policy[d_sum]
        
        if d_sum > 21:
            # dealer goes bust, win
            reward_list.append(1.0)
        else:
            # compare
            reward_list.append(cmp_r(p_sum, d_sum))

    assert len(state_list) == len(action_list) == len(reward_list) == t+1, (state_list, action_list, reward_list, t)
    return state_list, action_list, reward_list


def compute_return(rewards):
    Return = 0.0
    for reward in rewards:
        Return = DISCOUNT * Return + reward
    return Return

def compute_importance_ratio(states, b_actions, player_simple_policy):
    # player_simple_policy is fixed for estimation
    # Compute all important ratios in an eposide;
    # From state_t (from 0) -> T(t) - 1 (all in an eposide) to compute importance
    assert len(states) == len(b_actions)
    t_len = len(states)

    r_back = []
    r = 1.0
    for t_back in reversed(range(t_len)):
        b_action = b_actions[t_back]
        state = states[t_back]
        p_sum = state[0]
        t_action = player_simple_policy[p_sum]
        if b_action == t_action:
            # 1/0.5
            r *= 2.0
        else:
            # 0/0.5
            r *= 0.0
        r_back.append(r)
    
    return list(reversed(r_back))

def compute_IS_ratio(player_trajectory, behavior_policy_player, player_simple_policy):
    r_back = []
    r = 1.0
    for (usable_ace, player_sum, dealer_card), action in player_trajectory:
        t_action = player_simple_policy[player_sum]
        b_action = action
        if b_action == t_action:
            # 1/0.5
            r *= 2.0
        else:
            # 0/0.5
            r *= 0.0
        r_back.append(r)
    
    return list(reversed(r_back))

def SqErr(GT_Value, Est):
    diff = GT_Value - Est
    return diff * diff



def figure_5_3():

    # player stick_thres = 20
    player_simple_policy = ([None] + [1] * 19 + [0] * 11) # None is occupier, bust to 31
    # dealer stick_thres = 17
    dealer_simple_policy = ([None] + [1] * 16 + [0] * 14) # None is occupier, bust to 31

    start_states = [([1, 1, 1], 2, 1), ([1, 2], 2, 1)]
    initial_state = [True, 13, 2]

    # b_policy_prob = 0.5

    Runs = 100
    Episode_N = 10000

    IS_diff_mat = np.zeros((Runs, Episode_N))
    WIS_diff_mat = np.zeros((Runs, Episode_N))

    GT_State_Value = -0.27726

    state_t = 0 # always evaluate importance ratio of the first state 
    for run in range(Runs):
        W_Sum_V = 0.0
        count = 0.0
        w_sum = 0.0
        for episode_n in range(Episode_N):
            
            # states, actions, rewards = bj_ep_starting_state(start_states, dealer_simple_policy)
            # assert len(states) == len(actions) == len(rewards)
            # t_len = len(states)

            # # return of the first state
            # eposide_return = compute_return(rewards)

            # IS_ratios = compute_importance_ratio(states, actions, player_simple_policy)

            _, reward, player_trajectory = play(behavior_policy_player, initial_state=initial_state)
            eposide_return = reward
            IS_ratios = compute_IS_ratio(player_trajectory, behavior_policy_player, player_simple_policy)
            
            IS_ratio = IS_ratios[0]

            W_Sum_V += IS_ratio * eposide_return

            # Eq 5.6
            w_sum += IS_ratio
            if w_sum == 0.0:
                WIS_Est = 0.0
            else:
                WIS_Est = W_Sum_V/w_sum
            WIS_diff_mat[run, episode_n] = SqErr(GT_State_Value, WIS_Est)

            # Eq 5.5
            count += 1.0
            IS_Est = W_Sum_V/count
            IS_diff_mat[run, episode_n] = SqErr(GT_State_Value, IS_Est)

            print('R{} E{}: IS:{:.2f} WIS:{:.2f}'.format(run, episode_n, IS_Est, WIS_Est))
    pdb.set_trace()

    error_ordinary = np.mean(IS_diff_mat, axis=0)
    error_weighted = np.mean(WIS_diff_mat, axis=0)

    plt.plot(error_ordinary, label='Ordinary Importance Sampling')
    plt.plot(error_weighted, label='Weighted Importance Sampling')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Mean square error')
    plt.xscale('log')
    plt.legend()

    plt.savefig('./images/mine/figure_5_3.png')
    plt.close()

    


if __name__ == '__main__':
    figure_5_3()