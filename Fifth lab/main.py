import numpy as np
from scipy import linalg
from scipy.linalg import hilbert
import pandas as pd
from tabulate import tabulate


def cond_s(A: np.array) -> float:
    return linalg.norm(A) * linalg.norm(linalg.pinv(A))


def cond_v(A: np.array) -> float:
    ml = 1
    for i in range(len(A)):
        ml *= np.sqrt(A[i].T @ A[i])

    return ml / abs(linalg.det(A))


def cond_a(A: np.array) -> float:
    mx = float('-inf')
    inv = linalg.inv(A)
    for i in range(len(A)):
        val = linalg.norm(A[i]) * linalg.norm(inv[:, i])
        if mx < val:
            mx = val

    return mx


def get_all_cond(A: np.array) -> str:
    return f'Числа обусловленности: {cond_s(A)}, {cond_v(A)}, {cond_a(A)}'


def main():
    errors = [float(f'1e-0{i}') for i in range(2, 10)]

    A = np.array([[-401.0, 200.0], [1200.0, -601.0]])
    b1 = np.array([[200], [-600]])
    b2 = np.array([[199], [-601]])

    print(A)
    print(b1, '\n', b2)

    x1 = linalg.solve(A, b1)
    x2 = linalg.solve(A, b2)

    print(get_all_cond(A))
    print(f"|x1 - x2| / |x1|: ", linalg.norm(x1 - x2) / linalg.norm(x1))
    result = np.array([[0.0 for k in range(len(errors))] for j in range(2, 10)])

    for i in range(2, 10):
        print('-----------------------------')
        print(f'Матрица Гильберат порядка {i}')

        H = hilbert(i)
        b = np.random.rand(i, 1)
        x1 = linalg.solve(H, b)

        print(get_all_cond(H))
        values = []
        for err in reversed(errors):
            H = hilbert(i)
            H += (np.random.randint(3, size=[i, i]) - np.ones([i, i], dtype=int)) * err
            rand_signs = np.random.randint(3, size=i)
            rand_signs -= np.ones(i, dtype=int)
            b1 = b + err * rand_signs
            x2 = linalg.solve(H, b1)
            print(f"Варьируем на {err}, |x1 - x2| / |x1|: ", linalg.norm(x1 - x2) / linalg.norm(x1))
            values.append(linalg.norm(x1 - x2) / linalg.norm(x1))
        result[i - 2] = values

    print(tabulate(result, headers=[str(i) for i in range(2, 10)], showindex=reversed(errors)))
    print('----------------------------')
    diag = np.array([[5, 1, 0, 0],
                     [2, 7, 3, 0],
                     [0, 5, 23, 11],
                     [0, 0, 39, 50]], dtype=np.float64)

    print(diag)
    print(get_all_cond(diag))
    b1 = np.array([12.0, 18.0, 37.0, 41.0])
    b2 = b1 + np.random.normal(size=(1, 4))
    x1 = linalg.solve(diag, b1)
    x2 = linalg.solve(diag, b2[0])
    print(f"|x1 - x2| / |x1|: ", linalg.norm(x1 - x2) / linalg.norm(x1))


if __name__ == '__main__':
    main()
