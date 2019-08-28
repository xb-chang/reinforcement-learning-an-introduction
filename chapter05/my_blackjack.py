import numpy as np
from random import randint

DISCOUNT = 1.0

# A comprehensive state is (current sum, dealer's showning card, whether a usable ace hold)
# current sum < 12, always hit; activated states are [12-21]
# dealer's showing card: ACE - 10
# useable ace or not in player (keep changing every time)

# simple policy with a threshold
# This simple policy only cares the sum in hand.
# player stick_thres = 20
# dealer stick_thres = 17


def card_deck():
    # ACE - 10
    return randint(1, 10)


def figure_5_1():
    # simple policy with a threshold
    # This simple policy only cares the sum in hand.
    # player stick_thres = 20
    player_simple_policy = ([1] * 19 + [0] * 2)
    # dealer stick_thres = 17
    dealer_simple_policy = ([1] * 16 + [0] * 5)
    
    

if __name__ == '__main__':
    figure_5_1()
    # figure_5_2()
    # figure_5_3()