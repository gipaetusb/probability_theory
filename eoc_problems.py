import numpy as np
from itertools import combination, product

# 2.1
p1 = "3 x [4 or 3]"
answer = "YES"

# 2.2
p2 = 1 / 12 + 1 / 11 + 1 / 10 + 1 / 9
answer = "YES"

# 2.3
p3 = """
    [0,0,0,0],

    [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],

    [0,0,1,1],[1,1,0,0],[1,0,1,0],[0,1,0,1],

    [0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],

    [1,1,1,1]
"""
p3 = list(product((1,0), repeat=4))
p3 = np.sum(p3, axis=1)
answer = np.sum(p3==3)/len(p3) == np.sum(p3==2)/len(p3)
answer = "False: probability 3,1 > 2,2  (1/2 vs 3/8)"

# 2.4a
p4 = 3/len(list(product((1,2,3),repeat=3)))

# 2.5

