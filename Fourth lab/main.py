import numpy as np
from scipy import linalg


def QR(A: np.array):
    Q = np.array([[0] * len(A[0])] * len(A), dtype=np.float64)
    for i in range(len(A[0])):
        Q[:, i] = np.copy(A[:, i])
        for j in range(i):
            Q[:, i] -= Q[:, j] * (Q[:, j].T @ A[:, i])

        Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])
    R = (Q.T @ A)
    return Q, R


def cond_s(A: np.array):
    return linalg.norm(A) * linalg.norm(linalg.pinv(A))


def main():
    A_list = [np.array([[3, 5, 9], [2, 1, -3], [3, -5, 2], [2, 4, -1], [-1, 2, 8], [4, 1, 1]]),
              np.array([[1, 4214], [4274732, 7483]])]
    for A in A_list:
        Q, R = QR(A)
        print("A:")
        print(A)
        print("Q @ R:")
        print(Q @ R)
        print("Q:")
        print(Q)
        print("R: ")
        print(R)
        print("cond A: ", cond_s(A), ", cond Q: ", cond_s(Q), ", cond R: ", cond_s(R))
        print("-------------------------------------------------------------------")


if __name__ == '__main__':
    main()
