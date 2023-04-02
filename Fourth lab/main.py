import numpy as np


def QR(A: np.array):
    Q = np.array([[0] * len(A)] * len(A), dtype=np.float32)
    for i in range(len(A)):
        Q[:, i] = A[:, i]
        print(type(Q[0][0]))
        for j in range(i):
            Q[:, i] -= Q[:, j] * (Q[:, j].T @ A[:, i])

        Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])
        print(Q)
    R = (Q.T @ A)
    return Q, R


def cond(A: np.array):
    return np.sum(A**2)


def main():
    A = np.array([[1, 2], [3, 4]])
    Q, R = QR(A)
    print(Q, R)
    print(A, Q@R, cond(R) * cond(Q), cond(Q), cond(Q@R))


if __name__ == '__main__':
    main()
