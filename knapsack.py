import numpy as np
from tqdm import tqdm
from group import Group


class Knapsack:
    def __init__(self, weights, rules, goal):
        self.weights = weights
        self.rules = rules
        self.goal = goal
        assert len(self.weights) == len(self.rules)

        self.n_groups = len(self.weights)

        self.groups = [Group(self.weights[i], self.rules[i]) for i in range(self.n_groups)]

    def pack(self, n_iter):
        best_split = None
        best_loss = float('inf')
        for i in tqdm(range(n_iter)):
            subsplit = list()
            subsplit_sum = 0
            for group in self.groups:
                group_subsplit = group.gen_subsplit()
                subsplit_sum += group_subsplit.sum()
                subsplit.append(group_subsplit)
            loss = np.abs(subsplit_sum - self.goal)
            if loss < best_loss:
                best_loss = loss
                best_split = subsplit
                for g_i, group in enumerate(self.groups):
                    group.adjust_up(subsplit[g_i], i)
            elif loss == best_loss:
                continue
            else:
                for g_i, group in enumerate(self.groups):
                    group.adjust_down(subsplit[g_i], i)
        return best_split, best_loss


if __name__ == "__main__":
    w1 = [[1, 2, 3],
          [4, 5, 6],
          [1000, 9000, 100],
          [0.1, 0.00001, 0.5, 1, 10000, 1e-16, 1e8],
          np.arange(-10, 10).tolist()]
    w2 = [[1], [5], [0.1], [1e-6], [1e8]]
    w3 = [
        np.random.random(10).tolist(),
        np.random.randint(10, 1000, size=100).tolist(),
        np.random.beta(1, 2, size=30).tolist(),
        (np.random.random(10) / 10).tolist(),
        np.random.uniform(1000, 100000, size=20).tolist()
    ]
    w4 = [
        np.random.randint(0, 10, size=100).tolist(),
        np.random.randint(10, 1000, size=100).tolist(),
        np.random.beta(1, 2, size=100).tolist(),
        np.random.uniform(1000, 100000, size=100).tolist()
    ]

    rules = [[0, 1],
             [0, 1],
             [0, 1],
             [0, 1],
             [0, 1],
             ]
    limit1 = 3 + 6 + 100 + 0.5 + 7
    limit2 = 5.1
    limit3 = w3[0][5] + w3[1][50] + w3[2][15] + w3[3][5] + w3[4][10]
    limit4 = w4[0][50] + w4[1][50] + w4[2][50] + w4[3][50]

    ks = Knapsack(w4, [[0, 1]] * len(w4), limit4)
    best_split, best_loss = ks.pack(5000)

    print(f"Target weight: {limit3}")
    print(f"Best Loss: {best_loss}")
    print(f"Best split: {best_split}")
    print(f"Expected split: {[w4[0][50], w4[1][50], w4[2][50], w4[3][50]]}")
    print()
