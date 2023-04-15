import numpy as np
from scipy import linalg


def cond_s(A: np.array) -> float:
    return linalg.norm(A) * linalg.norm(linalg.pinv(A))


def cond_v(A: np.array) -> float:
    ...


def cond_a(A: np.array) -> float:
    ...


def main():
    ...


if __name__ == '__main__':
    main()
