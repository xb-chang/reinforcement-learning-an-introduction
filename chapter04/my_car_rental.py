import math
import numpy as np
from scipy.stats import poisson

import pdb

# Two Locations
L1_CAR_MAX=20
L2_CAR_MAX=20

# Earn / car rental
CAR_RENTAL = 10

# RENT Prob
L1_OUT_LAMB = 3
L2_OUT_LAMB = 4

# Return Prob
L1_IN_LAMB = 3
L2_IN_LAMB = 2

# prob = poisson.pmf(x, mu) P(X = x)
# prob = poisson.cdf(x, mu) P(X <= x)

MAX_CAR_MOVE = 5
# L1 -> L2: +; L2 -> L1: -
# Move Car Fees
CAR_MOVE_FEE = 2
# determinstic policies
policy = np.zeros((L1_CAR_MAX, L2_CAR_MAX))

# States are the #cars of two locates at the beginning of each day
# action: #cars move between them. L1 -> L2: +
# the net income is reward 


    

pdb.set_trace()