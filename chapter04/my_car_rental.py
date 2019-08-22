import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import math
import numpy as np

import pdb

# Two Locations
L1_CAR_MAX=20
L2_CAR_MAX=20
MAX_CARS = 20

# Earn / car rental
CAR_RENTAL = 10

# RENT Prob
L1_OUT_LAMB = 3
L2_OUT_LAMB = 4

# RENT_UP_BOUND = 20

# Return Prob
L1_IN_LAMB = 3
L2_IN_LAMB = 2

# RETURN_UP_BOUND = 20

# prob = poisson.pmf(x, mu) P(X = x)
# prob = poisson.cdf(x, mu) P(X <= x)

MAX_CAR_MOVE = 5
ACTION_LIST = range(-1*MAX_CAR_MOVE, MAX_CAR_MOVE+1)

# L1 -> L2: +; L2 -> L1: -
# Move Car Fees
CAR_MOVE_FEE = 2


DISCOUNT = 0.9

def rent(customers, cars, LAMB):
    '''
    Car renting models:
    #Customers;
    #Cars at a location (before rent);
    #poisson para;
    Return:
        #Cars rent out
        prob of this event
        #Cars left in a location
    '''
    if customers <= cars:
        out_num = customers
        remain_num = cars - out_num
        out_prob = poisson.pmf(out_num, LAMB)
    
    elif customers == cars + 1:
        # considering all cases with more customers than cars
        out_num = cars # all cars are rent out
        remain_num = 0 # 0 cars left
        out_prob = 1 - poisson.cdf(cars, LAMB) # prob sum of all cases

    else:
        raise ValueError('Customer: {}; Cars: {}'.format(customers, cars))
    
    assert MAX_CARS >= out_num >= 0
    assert MAX_CARS >= remain_num >= 0
    assert 0 <= out_prob <= 1
    return out_num, out_prob, remain_num



def return_car(customers, cars, LAMB, CAR_MAX):
    '''
    Car returning models:
    #customers: Cars return to location;
    #Cars at a location (before return);
    #poisson para;
    Return:
        prob of this event
        #Cars left in a location
    '''
    if customers + cars <= CAR_MAX:
        remain_num = customers + cars
        prob = poisson.pmf(customers, LAMB)
    elif customers + cars == CAR_MAX + 1:
        remain_num = CAR_MAX
        assert (customers-1) >= 0
        prob = 1 - poisson.cdf((customers-1), LAMB) # prob sum of all cases
    else:
        raise ValueError('customers {} + cars {} : {}'.format(customers, cars, customers + cars))
    
    assert 0 <= remain_num <= CAR_MAX
    assert 0 <= prob <= 1
    return prob, remain_num


def compute_dynamics(cars_L, L_OUT_LAMB, L_IN_LAMB, L_CAR_MAX):
    # with model, the dynamic is fixed from begining to before car movement via policy.
    L_temp_states = []
    for L_cutomers in range(cars_L+2):
        L_out, L_out_prob, rent_remain_cars_L = rent(L_cutomers, cars_L, L_OUT_LAMB)

        L_Return_Bound = L_CAR_MAX + 2 - rent_remain_cars_L
        for L_returns in range(L_Return_Bound):
            L_return_prob, remain_cars_L = return_car(L_returns, rent_remain_cars_L, L_IN_LAMB, L_CAR_MAX)

            L_temp_states.append((L_out, L_out_prob, L_return_prob, remain_cars_L))
    return L_temp_states

L1_Dynamics = {}
L2_Dynamics = {}

def Expected_Return_with_Mem(state, action, state_value):
    '''
    Policy Iteration is used. (Memory based) (less computing required)
    reward = Income - car movement fee
    States are the #cars of two locates at the end (not begining!!!) of each day 
    action: #cars move between them. L1 -> L2: +; [-5, 5]
    return:
        updated V(state)
    '''

    assert -5 <= action <= 5
    # move cars first, then one day starts
    # new states after processing and action
    # L1 -> L2: +

    cars_L1 = min(state[0] - action, L1_CAR_MAX)
    if cars_L1 in L1_Dynamics:
        L1_temp_states = L1_Dynamics[cars_L1]
    else:
        L1_temp_states = compute_dynamics(cars_L1, L1_OUT_LAMB, L1_IN_LAMB, L1_CAR_MAX)
        L1_Dynamics[cars_L1] = L1_temp_states

    cars_L2 = min(state[1] + action, L2_CAR_MAX)
    if cars_L2 in L2_Dynamics:
        L2_temp_states = L2_Dynamics[cars_L2]
    else:
        L2_temp_states = compute_dynamics(cars_L2, L2_OUT_LAMB, L2_IN_LAMB, L2_CAR_MAX)
        L2_Dynamics[cars_L2] = L2_temp_states
    
    assert L1_CAR_MAX >= cars_L1 >= 0 or L2_CAR_MAX >= cars_L2 >= 0

    value = 0.0

    for L1_out, L1_out_prob, L1_return_prob, remain_cars_L1 in L1_temp_states:
        for L2_out, L2_out_prob, L2_return_prob, remain_cars_L2 in L2_temp_states:
            Income = (L1_out + L2_out) * 10.0
            reward = Income - CAR_MOVE_FEE * abs(action)
            new_cars_L1 = remain_cars_L1
            new_cars_L2 = remain_cars_L2
            
            assert 0 <= new_cars_L1 <= 20 and  0 <= new_cars_L2 <= 20

            value += L1_out_prob*L1_return_prob*L2_out_prob*L2_return_prob * (reward + DISCOUNT * state_value[new_cars_L1][new_cars_L2])
    return value



def Expected_Return(state, action, state_value):
    '''
    Policy Iteration is used.
    reward = Income - car movement fee
    States are the #cars of two locates at the end (not begining!!!) of each day
    action: #cars move between them. L1 -> L2: +; [-5, 5]
    return:
        updated V(state)
    '''

    assert -5 <= action <= 5

    cars_L1 = min(state[0] - action, L1_CAR_MAX)
    cars_L2 = min(state[1] + action, L2_CAR_MAX)
    assert L1_CAR_MAX >= cars_L1 >= 0 or L2_CAR_MAX >= cars_L2 >= 0

    value = 0.0

    # cars_Lx + 1 means all cases with #cutomers >= #cars_Lx
    for L1_cutomers in range(cars_L1+2):
        L1_out, L1_out_prob, rent_remain_cars_L1 = rent(L1_cutomers, cars_L1, L1_OUT_LAMB)

        L1_Return_Bound = L1_CAR_MAX + 2 - rent_remain_cars_L1
        for L1_returns in range(L1_Return_Bound):
            L1_return_prob, remain_cars_L1 = return_car(L1_returns, rent_remain_cars_L1, L1_IN_LAMB, L1_CAR_MAX)

            for L2_cutomers in range(cars_L2+2):
                L2_out, L2_out_prob, rent_remain_cars_L2 = rent(L2_cutomers, cars_L2, L2_OUT_LAMB)
                
                L2_Return_Bound = L2_CAR_MAX + 2 - rent_remain_cars_L2
                for L2_returns in range(L2_Return_Bound):
                    L2_return_prob, remain_cars_L2 = return_car(L2_returns, rent_remain_cars_L2, L2_IN_LAMB, L2_CAR_MAX)

                    Income = (L1_out + L2_out) * 10.0
                    reward = Income - CAR_MOVE_FEE * abs(action)
                    new_cars_L1 = remain_cars_L1
                    new_cars_L2 = remain_cars_L2
                    
                    assert 0 <= new_cars_L1 <= 20 and  0 <= new_cars_L2 <= 20

                    value += L1_out_prob*L1_return_prob*L2_out_prob*L2_return_prob * (reward + DISCOUNT * state_value[new_cars_L1][new_cars_L2])

    return value

# Value iterations as well?

def figure_4_2_val_iter():
    # value iteration for final policy and values in one sweep
    # determinstic policies
    value = np.zeros((L1_CAR_MAX+1, L2_CAR_MAX+1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 0
    _, axes = plt.subplots(2, 1, figsize=(40, 20)) # a fig of policy and a fig of optimal values
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    while True:
        # value iteration, similar to policy improvement
        max_diff = 0
        for L1_cars in range(L1_CAR_MAX+1):
            for L2_cars in range(L2_CAR_MAX+1):
                old_value = value[L1_cars, L2_cars]
                action_values = []
                for action in ACTION_LIST:
                    if 0 <= action <= L1_cars or (-1*L2_cars) <= action <= 0: 
                        # action_values.append(Expected_Return([L1_cars, L2_cars], action, value))
                        action_values.append(Expected_Return_with_Mem([L1_cars, L2_cars], action, value))
                    else:
                        # not available actions
                        action_values.append(-np.inf)
                new_value = np.max(action_values)
                diff = abs(old_value - new_value)
                value[L1_cars, L2_cars] = new_value
                if max_diff < diff:
                    max_diff = diff
        iterations += 1
        print('Iter {}: max diff: {:.6f}'.format(iterations, max_diff))
        if max_diff < 1e-4:
            break
    
    # draw optimal values
    fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
    fig.set_ylabel('# cars at first location', fontsize=30)
    fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
    fig.set_xlabel('# cars at second location', fontsize=30)
    fig.set_title('optimal value', fontsize=30)

    # final optimal policy
    for L1_cars in range(L1_CAR_MAX+1):
        for L2_cars in range(L2_CAR_MAX+1):
            action_values = []
            for action in ACTION_LIST:
                if 0 <= action <= L1_cars or (-1*L2_cars) <= action <= 0: 
                    # action_values.append(Expected_Return([L1_cars, L2_cars], action, value))
                    action_values.append(Expected_Return_with_Mem([L1_cars, L2_cars], action, value))
                else:
                    # not available actions
                    action_values.append(-np.inf)
            new_action = ACTION_LIST[np.argmax(action_values)]
            policy[L1_cars, L2_cars] = new_action

    # draw optimal policy
    fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[0])
    fig.set_ylabel('# cars at first location', fontsize=30)
    fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
    fig.set_xlabel('# cars at second location', fontsize=30)
    fig.set_title('optimal policy'.format(0), fontsize=30)

    plt.savefig('./images/mine/figure_4_2_val_iter.png')
    plt.close()


def figure_4_2():
    # determinstic policies
    value = np.zeros((L1_CAR_MAX+1, L2_CAR_MAX+1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # policy evaluation (in-place)
        eval_iter = 0
        while True:
            old_value = value.copy()
            for L1_cars in range(L1_CAR_MAX+1):
                for L2_cars in range(L2_CAR_MAX+1):
                    action = policy[L1_cars, L2_cars]
                    # value[L1_cars, L2_cars] = Expected_Return([L1_cars, L2_cars], action, value)
                    value[L1_cars, L2_cars] = Expected_Return_with_Mem([L1_cars, L2_cars], action, value)
                    # print([L1_cars, L2_cars])
            
            max_value_change = abs(old_value - value).max()
            eval_iter += 1
            print('{} max value change {}'.format(eval_iter, max_value_change))
            if max_value_change < 1e-4:
            # if max_value_change < 1e-2:
                break

        # policy improvement
        policy_stable = True
        for L1_cars in range(L1_CAR_MAX+1):
            for L2_cars in range(L2_CAR_MAX+1):
                old_action = policy[L1_cars, L2_cars]
                action_values = []
                for action in ACTION_LIST:
                    if 0 <= action <= L1_cars or (-1*L2_cars) <= action <= 0: 
                        # action_values.append(Expected_Return([L1_cars, L2_cars], action, value))
                        action_values.append(Expected_Return_with_Mem([L1_cars, L2_cars], action, value))
                    else:
                        # not available actions
                        action_values.append(-np.inf)
                new_action = ACTION_LIST[np.argmax(action_values)]
                policy[L1_cars, L2_cars] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break

        iterations += 1

    plt.savefig('./images/mine/figure_4_2.png')
    plt.close()



if __name__ == '__main__':
    figure_4_2()
    figure_4_2_val_iter()
