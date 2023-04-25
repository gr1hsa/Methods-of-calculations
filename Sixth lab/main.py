import numpy as np
from scipy import linalg
from scipy.linalg import eigh
import matplotlib.pyplot as plt


def power_method(A: np.array, epsilon=1e-10):
    '''
    :param A:
    :param epsilon:
    :return: eigenval_curr, x_curr, steps
    '''
    x_prev = np.ones_like(A[0])
    x_curr = np.ones_like(A[0])
    eigenval_prev = float('-inf')
    steps = 0

    while True:
        steps += 1
        x_curr = A @ x_prev
        eigenval_curr = x_curr[0] / x_prev[0]
        if abs(eigenval_curr - eigenval_prev) < epsilon:
            break

        if steps % 10 == 0:
            x_curr /= linalg.norm(x_curr)

        x_prev = x_curr
        eigenval_prev = eigenval_curr

    return eigenval_curr, x_curr / linalg.norm(x_curr), steps


def scalar_method(A: np.array, epsilon=1e-10):
    x_prev = np.ones_like(A[0])
    y_prev = np.ones_like(A[0])
    eigenval_prev = float('-inf')
    steps = 0
    A_T = A.T

    while True:
        steps += 1
        x_curr = A @ x_prev
        y_curr = A_T @ y_prev

        eigenval_curr = (x_curr @ y_curr.T) / (x_prev @ y_curr.T)

        if abs(eigenval_curr - eigenval_prev) < epsilon:
            break

        if steps % 10 == 0:
            x_curr /= linalg.norm(x_curr)
            y_curr /= linalg.norm(y_curr)

        x_prev = x_curr
        y_prev = y_curr
        eigenval_prev = eigenval_curr

    return eigenval_curr, x_curr / linalg.norm(x_curr), steps


def main():
    A = np.array([[-0.81417, -0.01937, 0.41372],
                  [-0.01937, 0.54414, 0.00590],
                  [0.41372, 0.00590, -0.81445]])

    errors = [1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10, 1e-11, 1e-12]
    steps_pow, steps_scal = [], []
    for er in errors:
        steps_pow.append(power_method(A, epsilon=er)[2])
        steps_scal.append(scalar_method(A, epsilon=er)[2])

    pow_eigenval, pow_eigenvec, pow_steps = power_method(A, epsilon=1e-14)
    scal_eigenval, scal_eigenvec, scal_steps = scalar_method(A, epsilon=1e-14)

    print(f"Степенной метод. с.ч.: {pow_eigenval}, с.в.: {pow_eigenvec}, шагов: {pow_steps}")
    print(f"Скалярный метод. с.ч.: {scal_eigenval}, с.в.: {scal_eigenvec}, шагов: {scal_steps}")
    print(f'Библиотечное решение: с.ч.: {max(eigh(A)[0], key=lambda x: abs(x))},'
          f' с.в.: {eigh(A)[1].T[np.argmax(np.abs(eigh(A)[0]))]} ')

    plt.plot([i + 2 for i in range(len(errors))], steps_pow,
             color='red', label='power')
    plt.plot([i + 2 for i in range(len(errors))], steps_scal,
             color='blue', label='scalar')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
