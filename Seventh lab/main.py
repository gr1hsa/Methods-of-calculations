import numpy as np
from math import sqrt
from scipy.linalg import hilbert
from numpy.linalg import eigvals
import matplotlib.pyplot as plt



def get_upper_triangle(A: np.array):
    A_ret = np.zeros_like(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i < j:
                A_ret[i][j] = A[i][j]

    return A_ret


def jacobi_method(A: np.array, method='mx', epsilon=1e-5):
    A_mx = np.copy(A)
    steps = 0
    elems = [(i, j) for i in range(len(A)) for j in range(len(A)) if i != j]

    while True:
        A_mx_new = np.copy(A_mx)
        steps += 1
        upper_triangle = get_upper_triangle(A_mx)
        if method == 'mx':
            i, j = np.unravel_index(np.argmax(np.triu(abs(upper_triangle), k=1), axis=None), upper_triangle.shape)

        elif method == 'ord':
            i, j = elems[steps % (len(A) * (len(A) - 1))]

        #i_mx, j_mx = np.unravel_index(np.argmax(np.triu(abs(upper_triangle), k=1), axis=None), upper_triangle.shape)

        # if abs(A_mx[i_mx][j_mx]) < epsilon or i_mx == j_mx:
        #     break

        if get_R_max(A_mx) < epsilon:
            break

        d = sqrt((A_mx[i][i] - A_mx[j][j]) ** 2 + 4 * ((A_mx[i][j]) ** 2))
        c = sqrt((1 + abs(A_mx[i][i] - A_mx[j][j]) / d) / 2)
        s = np.sign(A_mx[i][j]*(A_mx[i][i] - A_mx[j][j])) * sqrt((1 - (abs(A_mx[i][i] - A_mx[j][j]) / d)) / 2)

        for k in range(len(A_mx)):
            if k != i and k != j:
                A_mx_new[k][i] = c * A_mx[k][i] + s * A_mx[k][j]
                A_mx_new[i][k] = c * A_mx[k][i] + s * A_mx[k][j]
                A_mx_new[j][k] = -s * A_mx[k][i] + c * A_mx[k][j]
                A_mx_new[k][j] = -s * A_mx[k][i] + c * A_mx[k][j]

        A_mx_new[i][i] = c * c * A_mx[i][i] + 2 * c * s * A_mx[i][j] + s * s * A_mx[j][j]
        A_mx_new[j][j] = s * s * A_mx[i][i] - 2 * c * s * A_mx[i][j] + c * c * A_mx[j][j]
        A_mx_new[i][j] = 0
        A_mx_new[j][i] = 0
        A_mx = np.copy(A_mx_new)

    return np.diag(A_mx), steps


def get_gershgorin_estimate(A: np.array):
    regions = []
    for i in range(len(A)):
        sm = 0
        for j in range(len(A[0])):
            if i != j:
                sm += abs(A[i][j])
        regions.append((A[i][i] - sm, A[i][i] + sm))

    return min(regions, key=lambda x: x[0])[0],  max(regions, key=lambda x: x[1])[1]


def get_R_max(A: np.array):
    mx = float('-inf')
    for i in range(len(A)):
        if mx < sum(np.abs(A[i])) - abs(A[i][i]):
            mx = sum(np.abs(A[i])) - abs(A[i][i])

    return mx


def main():
    A = np.array([[-0.81417, -0.01937, 0.41372], [-0.01937, 0.54414, 0.00590], [0.41372, 0.00590, -0.81445]])
    print(A)
    print("Оценка по теореме Гешгорина: \n", get_gershgorin_estimate(A))
    matrix_mx, steps_mx = jacobi_method(A, epsilon=1e-10)
    matrix_ord, steps_ord = jacobi_method(A, 'ord', epsilon=1e-10)
    print("Приближённые занчения, метод максимального элемента:\n",
          sorted(matrix_mx), " Количество шагов: ", steps_mx,  '\n',
          "Метод взятия элементов по порядку:\n", sorted(matrix_ord), " Количество шагов: ", steps_ord)
    print("Библиотечное решение:\n", sorted(eigvals(A)))

    while True:
        n = int(input("Введите размер матрица Гильбера: "))
        A = hilbert(n)
        print("Оценка по теореме Гешгорина: \n", get_gershgorin_estimate(A))
        matrix_mx, steps_mx = jacobi_method(A, epsilon=1e-10)
        matrix_ord, steps_ord = jacobi_method(A, 'ord', epsilon=1e-10)
        print("Приближённые занчения, метод максимального элемента:\n",
              sorted(matrix_mx), " Количество шагов: ", steps_mx, '\n',
              "Метод взятия элементов по порядку:\n", sorted(matrix_ord), " Количество шагов: ", steps_ord)
        print("Библиотечное решение:\n", sorted(eigvals(A)))

        errors = [1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10]
        steps = []
        for er in errors:
            steps.append((jacobi_method(A, epsilon=er)[1], jacobi_method(A, 'ord', epsilon=er)[1]))

        plt.plot([i + 2 for i in range(len(errors))], [steps[i][0] for i in range(len(steps))],
                 color='red', label='max')
        plt.plot([i + 2 for i in range(len(errors))], [steps[i][1] for i in range(len(steps))],
                 color='blue', label='ord')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
