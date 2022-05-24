import numpy as np


def random_stochastic_vector(n: int, left_bound=0, right_bound=999, seed=None):
    """
    Функция создания стахостического вектора из полиномиального распределения с количество элементов <n>.
    Parameters
    ----------
    n : int
        Количество элементов для полиномиальной схемы
    left_bound : int
        Левая граница
    right_bound : int
        Правая граница
    seed : optional
        Зерно ГПСЧ
    Returns
    -------
    np.ndarray
        Стахостический вектор
    """
    rng = np.random.default_rng(seed=seed)
    random_int_vec = np.zeros(n)
    for i in range(n):
        random_int_vec[i] = rng.integers(left_bound, right_bound)
    _sum = random_int_vec.sum()
    return random_int_vec / _sum


def generate_multinomial_rv(p_vec):
    cum_sum = np.append(np.array([0]), np.cumsum(p_vec))
    u = np.random.random()
    rv = -1
    for j in range(cum_sum.shape[0] - 1):
        if j == 0 and cum_sum[j] <= u <= cum_sum[j + 1]:
            rv = j
            break
        elif cum_sum[j] < u <= cum_sum[j + 1]:
            rv = j
            break
    return rv


class ProbabilitySplit:
    def __init__(self, size: int, can_be_empty: bool):
        self.rng = np.random.default_rng()

        # if can_be_empty:
            # last item in vector shows empty choice
            # size += 1
        self.can_be_empty = can_be_empty
        self.size = size
        self.probs = random_stochastic_vector(size)

    def generate_subsplit(self, size):
        subsplit = set()

        for i in range(size):
            rv = generate_multinomial_rv(self.probs)
            # if rv == self.size - 1 and self.can_be_empty:
            #     continue
            while rv in subsplit:
                rv = generate_multinomial_rv(self.probs)
            subsplit.add(rv)

        return np.array(sorted(subsplit))

    def adjust_up(self, idx, iteration):
        delta = (1 - self.probs[idx]) * self.rng.random()
        mem_val = self.probs[idx] + delta
        assert 0 <= mem_val <= 1

        self.probs[idx] = 0

        if self.probs.size == 1 and self.probs.sum() == 0:
            vec_ratios = np.array([1])
        else:
            vec_ratios = self.probs / self.probs.sum()
        self.probs -= vec_ratios * delta

        self.probs[idx] = mem_val

        assert np.abs(self.probs.sum() - 1) < 1e-6

    def adjust_down(self, idx, iteration):
        delta = self.probs[idx] * self.rng.random()
        mem_val = self.probs[idx] - delta
        assert 0 <= mem_val <= 1

        self.probs[idx] = 0

        if self.probs.size == 1 and self.probs.sum() == 0:
            vec_ratios = np.array([1])
        else:
            vec_ratios = self.probs / self.probs.sum()
        self.probs += vec_ratios * delta

        self.probs[idx] += mem_val

        assert np.abs(self.probs.sum() - 1) < 1e-6
