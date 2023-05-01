import numpy as np
from math import sin
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import linalg


def f(x: float, t: float) -> float:
    return x ** 3 + t ** 3


def explicit_schema(num_x: int, num_time: int, x_linspace: list[float], t_linspace: list[float],
                    tau: float, h: float, c: float, mu, mu1, mu2, f):
    u = np.zeros((num_x, num_time))

    for i in range(num_x):
        u[i][0] = mu(x_linspace[i])

    for k in range(num_time):
        u[0][k] = mu1(t_linspace[k])
        u[-1][k] = mu2(t_linspace[k])

    for k in range(1, num_time):
        for i in range(1, num_x - 1):
            u[i][k] = tau * c * (u[i - 1][k - 1] - 2 * u[i][k - 1] + u[i + 1][k - 1]) / (h ** 2) \
                      + tau * f(x_linspace[i], t_linspace[k - 1]) + u[i][k - 1]

    return u


def implicit_schema(num_x: int, num_time: int, x_linspace: list[float], t_linspace: list[float],
                    tau: float, h: float, c: float, mu, mu1, mu2, f):
    u = np.zeros((num_x, num_time), dtype=np.float64)

    for i in range(num_x):
        u[i][0] = mu(x_linspace[i])

    for k in range(num_time):
        u[0][k] = mu1(t_linspace[k])
        u[-1][k] = mu2(t_linspace[k])

    A = np.zeros((3, num_x - 2), dtype=np.float64)

    cf = c * tau / (h ** 2)

    A[0, 1:] = -cf
    A[1, :] = 1 + 2 * cf
    A[2, :-1] = -cf

    for k in range(1, num_time):
        b = np.zeros(num_x - 2, dtype=np.float64)
        b[0] = u[1][k - 1] + cf * u[0][k] + tau * f(x_linspace[1], t_linspace[k])

        for i in range(1, num_x - 3):
            b[i] = tau * f(x_linspace[i + 1], t_linspace[k]) + u[i + 1][k - 1]

        b[-1] = u[-2][k - 1] + cf * u[-1][k] + tau * f(x_linspace[-2], t_linspace[k])
        u[1:-1, k] = linalg.solve_banded((1, 1), A, b)

    return u


def test(explicit=True, bad=False):
    a = 1

    mu1 = lambda t: sin(t)
    mu2 = lambda t: sin(t)
    mu = lambda x: sin(x)

    num_x = 20
    num_time = 100
    if bad:
        num_x = 50

    x_linspace, h = np.linspace(0, 3.14, num_x, retstep=True)
    t_linspace, tau = np.linspace(0, 1, num_time, retstep=True)

    print(f"(h ** 2) / (2 * a) - tau = {(h ** 2) / (2 * a) - tau}")

    if explicit:
        U = explicit_schema(num_x, num_time, x_linspace, t_linspace, tau, h, a, mu, mu1, mu2, f)
    else:
        U = implicit_schema(num_x, num_time, x_linspace, t_linspace, tau, h, a, mu, mu1, mu2, f)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X, T = np.meshgrid(x_linspace, t_linspace, indexing='ij')
    ax.set_ylabel('time')
    ax.set_xlabel('X')

    surf = ax.plot_surface(X, T, U, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def main():

    test()
    test(explicit=False)
    test(bad=True)
    test(explicit=False, bad=True)


if __name__ == '__main__':
    main()
