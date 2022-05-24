import numpy as np
from typing import List
from prob_split import ProbabilitySplit


class Group:
    def __init__(self, split: np.ndarray, rule: List):
        self.split = np.array(split)
        self.rule = np.array(rule)

        self.prob_split = ProbabilitySplit(self.split.size, False)

        self.c_split = np.array(self.rule)
        self.c_probs = ProbabilitySplit(len(rule), False)

    def gen_subsplit(self):
        c_idx = self.c_probs.generate_subsplit(1)[0]
        c_val = self.c_split[c_idx]

        rv_split = self.prob_split.generate_subsplit(c_val)
        if rv_split.size == 0:
            return np.array([])
        return self.split[rv_split]

    def adjust_up(self, subsplit, iteration):
        c_val = subsplit.size
        c_idx = np.where(self.c_split == c_val)[0][0]

        self.c_probs.adjust_up(c_idx, iteration)
        for s_val in subsplit:
            idx = np.where(self.split == s_val)[0][0]
            self.prob_split.adjust_up(idx, iteration)

    def adjust_down(self, subsplit, iteration):
        c_val = subsplit.size
        c_idx = np.where(self.c_split == c_val)[0][0]

        self.c_probs.adjust_down(c_idx, iteration)
        for s_val in subsplit:
            idx = np.where(self.split == s_val)[0][0]
            self.prob_split.adjust_down(idx, iteration)
