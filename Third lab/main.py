from math import cos, pi, factorial
import numpy as np
from scipy import integrate
from scipy.special import jacobi
from numpy.linalg import solve
from matplotlib import pyplot as plt


def p(x: float) -> float:
    return 1 / (2 + x)


def r(x: float) -> float:
    return cos(x)


def f(x: float) -> float:
    return 1 + x


def w(x: float) -> float:
    return 1 - x ** 2


def G(n, k, a, b):
    return factorial(a + b + n + k + 1) / (2 ** k * factorial(a + b + n + 1))


def jacobi_polynoms(n: int, k=1) -> list:
    polynoms = [lambda x: 1, lambda x: (k + 1) * x]
    result = []
    for j in range(2, n):
        polynoms.append(lambda x: ((n + k + 2) * (2 * n + 2 * k + 3) * x * polynoms[j - 1](x) -
                                    (n + k + 2) * (n + k + 1) * polynoms[j - 2](x)) /
                                    ((n + 2 * k + 2) * (n + 2)))
    for i in range(len(polynoms)):
        result.append(lambda x: w(x) * polynoms[i](x))
    return result


def jacobi_diff(n: int, k=1) -> list:
    polynoms = jacobi_polynoms(n+1, k-1)[1:]
    result = []
    for i in range(len(polynoms)):
        result.append(lambda x: -2 * (n + 1) * (1 - x) ** (k - 1) * polynoms[i](x))

    return result


def ritz_method(n: int, a=-1, b=1, k=1):
    polynoms = [jacobi(i, k, k) for i in range(n)]
    polynoms_diff = [jacobi(i + 1, k-1, k-1) for i in range(n)]
    A = np.zeros([n, n])
    vals = np.zeros([n])
    for i in range(n):
        for j in range(n):
            A[i][j] = integrate.quad(lambda x: p(x) * polynoms_diff[i](x) * polynoms_diff[j](x) *
                                           (-2) * (i + 1) * w(x) ** (k - 1) *
                                            (-2) * (j + 1) * w(x) ** (k - 1) +
                                            r(x) * polynoms[i](x) * polynoms[j](x) * w(x) ** 2, a, b)[0]
        vals[i] = integrate.quad(lambda x: f(x) * polynoms[i](x) * w(x), a, b)[0]

    x = solve(A, vals)
    return lambda z: sum([x[i] * polynoms[i](z) * w(z) for i in range(n)])


def collocation_method(n: int, b=1, a=-1, k=1):
    h = (b - a) / n
    dots_cheb = [cos((2 * i - 1) * pi / (2 * n)) for i in range(1, n + 1)]
    dots_linspace = [a + h * i for i in range(n)]
    polynoms = []
    polynoms_diff = [lambda x: -2 * x * jacobi(0, k, k)(x)]
    polynoms_diff_diff = [lambda x: -2 * jacobi(0, k, k)(x), lambda x: -2 * jacobi(1, k, k)(x) - 4 * x * jacobi(0, k+1, k+1)(x)]
    for i in range(n):
        def funcC(i):
            def func(z): return (1 - z ** 2) * jacobi(i, 1, 1)(z)
            return func
        polynoms.append(funcC(i))
    for i in range(1, n):
        def funcC(i):
            def func(z): return -2 * z * jacobi(i, k, k)(z) + (1 - z ** 2) * jacobi(i - 1, k+1, k+1)(z)
            return func
        polynoms_diff.append(funcC(i))
    for i in range(2, n):
        def funcC(i):
            def func(z): return -2 * jacobi(i, k, k)(z) - 4 * z * jacobi(i-1, k+1, k+1)(z) + (1 - z ** 2) * jacobi(i-2, k+2, k+2)(z)
            return func
        polynoms_diff_diff.append(funcC(i))


    A_cheb = np.zeros([n,n])
    A_lin = np.zeros([n, n])
    vals_cheb = np.zeros([n])
    vals_lin = np.zeros([n])

    for i in range(n):
        x1 = dots_cheb[i]
        x2 = dots_linspace[i]
        for j in range(n):
            A_cheb[i][j] = -polynoms_diff_diff[j](x1) / (x1 + 2) + polynoms_diff[j](x1) / (x1 + 2) ** 2 + cos(x1) * polynoms[j](x1)
            A_lin[i][j] = -polynoms_diff_diff[j](x2) / (x2 + 2) + polynoms_diff[j](x2) / (x2 + 2) ** 2 + cos(x2) * polynoms[j](x2)
        vals_cheb[i] = f(x1)
        vals_lin[i] = f(x2)

    x1 = solve(A_cheb, vals_cheb)
    x2 = solve(A_lin, vals_lin)

    def func1(z): return sum(x1[i] * polynoms[i](z) for i in range(len(polynoms)))

    def func2(z):
        return sum(x2[i] * polynoms[i](z) for i in range(len(polynoms)))

    return (func1, func2)


def main():
    while True:
        n = int(input("Введите n: "))
        solve_cheb, solve_lin = collocation_method(n)
        solve_ritz = ritz_method(n)
        x = np.linspace(-1, 1, 1000)

        plt.plot(x, solve_ritz(x), color='red', label=f'ritz')
        plt.plot(x, solve_cheb(x), color='blue', label='collocation чебышев')
        plt.plot(x, solve_lin(x), color='yellow', label='collocation равностоящие')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
