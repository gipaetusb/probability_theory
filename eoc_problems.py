import math
import numpy as np
from itertools import product, permutations

# 2.1
p1 = "3 x [4 or 3]"
answer = "YES"

# 2.2
p2 = 1 / 12 + 1 / 11 + 1 / 10 + 1 / 9
answer = "YES"

# 2.3
p2_3 = """
    [0,0,0,0],

    [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],

    [0,0,1,1],[1,1,0,0],[1,0,1,0],[0,1,0,1],

    [0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],

    [1,1,1,1]
"""
p3 = list(product((1, 0), repeat=4))
p3 = np.sum(p3, axis=1)
answer2_3 = np.sum(p3 == 3) / len(p3) == np.sum(p3 == 2) / len(p3)
answer = "False: probability 3,1 > 2,2  (1/2 vs 3/8)"

# 2.4a
p4 = 3 / len(list(product((1, 2, 3), repeat=3)))

# 2.5
p = np.array(permutations(range(10)))
solution = np.sum(np.where(p == 0, 1, 0), axis=0) / len(p)

solution = "Doesn't change anything"

# 2.7
manque = [
    (1, 1),
    (1, 2), (2, 1),
    (2, 2), (3, 1), (1, 3),
    (2, 3), (3, 2), (4, 1), (1, 4),
    (3, 3), (4, 2), (2, 4), (5, 1), (1, 5),
]

seven = [
    (3, 4), (4, 3), (5, 2), (2, 5), (6, 1), (1, 6),
]

passe = [
    (4, 4), (5, 3), (3, 5), (6, 2), (2, 6),
    (4, 5), (5, 4), (6, 3), (3, 6),
    (5, 5), (6, 4), (4, 6),
    (5, 6), (6, 5),
    (6, 6),
]

chances = len(manque) + len(seven) + len(passe)  # two dice = (1/6)**2

bet_manque = 2 * len(manque) / chances + 0 * len(seven + passe) / chances
bet_passe = 2 * len(passe) / chances + 0 * len(seven + manque) / chances
print(bet_manque == bet_passe)

# 2.9
r = 0.05
f = 1.25
p = 0.9
α = (p * f - (1 + r)) / (f - (1 + r))
G_inf = p * np.log((1 - α) * (1 + r) + α * f) + (1 - p) * np.log((1 - α) * (1 + r))
long_run_rate = np.exp(G_inf) - 1  # since Vn = V0 * e**(nG)

# 2.11
ppl = 25
r = 10  # NOTE: if we tell 10 times, we'll have 11 people playin, not 10
# not repeated ?
n = ppl - 1
"[n / n] * [(n - 1) * ... * (n - 9) / (n - 1)]"

p2_11: float = 1
for i in range(1, 10):  # NOTE: this accounts for 9 iters
    p2_11 = p2_11 * (n - i) / (n - 1)

p2_11_with_factorials = (math.factorial(n - 1) / math.factorial(n - 1 - 9)) / ((n - 1)**9)

# not at start
"""
The first and the second are alright
Then each one can tell to 23 other people, so the chance of not being the originator is (23-1)/23
1 * 1 * 22/23 * 22/23 * 22/23 ...repeat (11-2) times
"""
he_says = ((n - 2) / (n - 1))**(r - 1)

# exp number ppl reached (use simulation)?
stat = 0
iters = 10**4
for it in range(iters):
    people = np.arange(25)
    reached = np.array([0])
    teller = 0
    for i in range(10):
        if i > 1:
            teller = reached[-2]
        current = reached[-1]
        candidates = people[~np.isin(people, [teller, current])]
        reached = np.append(
            reached,
            candidates[np.random.randint(low=0, high=len(candidates))]
        )
    stat += len(np.unique(reached))
print(stat / iters)


# 2.13
"""
If reveals left beaker first: L-Odd / R-Even; L-Odd / R-Odd; L-Even / R-Odd; L-Even / R-Even
If reveals right beaker first: R-Odd / L-even; R-Odd / L-Odd; R-Even / L-Odd; R-Even / R-Even

So, sample space:
L-E / R-O; L-E / R-E;
R-E / L-O; R-E / L-E;

space = 4 - 1
chance left  = 1 / 3
"""

# 2.15
s215 = 10**8
secs_h = 60**2
newspaper = np.random.choice(np.arange(secs_h), size=s215)
mrjohnson = np.random.choice(np.arange(secs_h / 2, 3 * secs_h / 2), size=s215)
print(np.sum(newspaper < mrjohnson) / s215)  # 87.5%


# 2.17
def prob_f(a: np.array, b: np.array, c: np.array):
    """
    Use quadratic formula: x = (-B ± √(B^2 - 4AC)) / 2A

    For a solution to exist it must hold B^2 - 4AC >= 0
    """
    return np.sum(b**2 - 4 * a * c >= 0) / len(a)


rep = 10**7
print("Any number within |q|")
for q in (1, 10, 100, 1000, 10000):
    a, b, c = np.random.uniform(low=-q, high=q, size=(3, rep))
    prob = prob_f(a, b, c)
    print(f"For q={q}, P={round(prob*100, 1)}%")

print("Non-zero integers")
for q in (1, 10, 100, 1000, 10000):
    univ = np.arange(start=-q, stop=q + 1, step=1)
    univ = univ[univ != 0]
    a, b, c = np.random.choice(univ, size=(3, rep))
    prob = prob_f(a, b, c)
    print(f"For q={q}, P={round(prob*100, 1)}%")

# 2.19
"""
unit circle: X^2 + Y^2 = 1
"""
