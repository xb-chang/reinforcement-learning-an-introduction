import numpy as np
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

    state_list = []
    state_list.append((p_sum, dealer_show, ace_heat))

    action_list = []
    action = p_policy[p_sum]
    action_list.append(action)
    
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
    U_ACE_V = np.zeros((10, 10))
    # no usable ace state
    N_ACE_V = np.zeros((10, 10))
    # 200 states in total (no/usable ACE)
    # V[0]: No Usable Ace
    # V[1]: Usable Ace
    V = [N_ACE_V, U_ACE_V]
    
    while True:
        # state: (p_sum, d_show, u_ace: 0-no; 1-yes), p_sum can < 12 or bust
        # action: stick-0; hit-1
        # reward: -1, 0, 1
        states, actions, rewards = bj_eposide_gen(player_simple_policy, dealer_simple_policy)
        # print(states, rewards)
        # pdb.set_trace()
        
    

if __name__ == '__main__':
    figure_5_1()
    # figure_5_2()
    # figure_5_3()