import numpy as np


def solve_itteration(A: np.array, b: np.array, epsilon: float) -> np.array:
    #Если будут ошибочки, то нужно будет следить, чтобы на диагонали не было 0
    B = np.zeros_like(A)
    betta = np.zeros_like(b)
    for i in range(len(A)):
        betta[i][0] = b[i][0] / A[i][i]
        for j in range(len(A[0])):
            if i == j:
                B[i][j] = 0
            else:
                B[i][j] = -A[i][j] / A[i][i]
    x = np.zeros_like(betta)
    steps = 0
    while True:
        steps += 1
        x_new = B @ x + betta
        if np.sum(np.abs(x - x_new)) < epsilon:
            break
        x = np.copy(x_new)
    return (x_new, np.sum(np.abs(x - x_new)), steps)


def seidel_method(A: np.array, b: np.array, epsilon: float, w=1):
    if w == 0:
        raise Exception("w не может быть равен 0")

    x = np.zeros_like(b)
    x_new = np.zeros_like(x)
    steps = 0
    while True:
        steps += 1
        for i in range(len(A)):
            x_new[i] = ((-w/A[i][i]) * ((A[i][0:i] @ x_new[0:i]) + (A[i][i+1:len(A)] @ x[i+1:len(A)]) - b[i][0]) -
                           (w - 1) * x[i][0])

        if np.sum(np.abs(x - x_new)) < epsilon:
            break
        x = np.copy(x_new)
    return (x_new, np.sum(np.abs(x - x_new)), steps)


def generate_matrix(matrix_dimension: int) -> tuple:
    """

    Args:
        matrix_dimension (int): _description_

    Returns:
        tuple: _description_
    """
    # A = np.random.normal(0, 1, (matrix_dimension, matrix_dimension))
    # x = np.random.normal(0, 1, (matrix_dimension, 1))
    A = np.random.random((matrix_dimension, matrix_dimension))
    x = np.random.random((matrix_dimension, 1))
    for i in range(len(A)):
        A[i][i] += -np.sign(A[i][i])*(2*np.abs(A[i][i]) - np.sum(np.abs(A[i]))) + 1 * np.sign(A[i][i])
    b = A @ x
    return (A, x, b)


def main():
    n = 10
    epsilon = 1e-3
    A = None
    while True:
        n = int(input(f"Введите размерность матрицы или enter для {n=}: ") or n)
        # if A is not None:
        #     ans = input(f"Сгенерировать новую матрицу? y/n")
        #     if ans == y:
        #         pass
        epsilon = float(input(f"Введите размерность матрицы или enter для {epsilon=}: ") or epsilon)
        A, x, b = generate_matrix(n)
        x_iter, error_iter, steps_iter = solve_itteration(A, b, epsilon)
        x_seidel, error_seidel, steps_seidel = seidel_method(A, b, epsilon, 1)
        print(f"Метод простой итерации\nКоличетсво шагов: {steps_iter}, оценка ошибки: {error_iter}, "
              f"ошибка на ответе: {np.sum(np.abs(x_iter - x))}")
        print(f"Метод Зейделя\nКоличетсво шагов: {steps_seidel}, оценка ошибки: {error_seidel}, "
              f"ошибка на ответе: {np.sum(np.abs(x_seidel - x))}")


if __name__ == '__main__':
    main()
