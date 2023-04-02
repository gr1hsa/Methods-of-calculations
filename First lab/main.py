from math import cos, log, sin
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt


# def p(x: float) -> float:
#     return (6 + x) / (7 + 3 * x)
#
#
# def q(x: float) -> float:
#     return x / 2 - 1
#
#
# def r(x: float) -> float:
#     return 1 + 0.5 * cos(x)
#
#
# def f(x: float) -> float:
#     return 1 - x / 3
# def p(x):
#     return (5 - x) / (7 - 3 * x)
#
#
# def q(x):
#     return - (1 - x) / 2
#
#
# def r(x):
#     return 1 + 1 / 2 * sin(x)
#
#
# def f(x):
#     return 1 / 2 + x / 2

def p(x):
    return 1 / (2 + x)


def q(x):
    return 0


def r(x):
    return cos(x)


def f(x):
    return 1 + x


def main():
    print("Вариант 9")
    print("Дифференциальное уравнение: -u''(6 + x) / ( 7 + 3x) - u'(1 - x / 2) + u(1 + 0.5cos(x)) = 1 - x/3")
    mode = 2
    while True:
        n = 5
        a = -1
        b = 1
        h = (b - a) / n
        alpha1 = 1
        alpha2 = 0
        betta1 = 1
        betta2 = 0
        alpha = 0
        betta = 0
        solves = []
        dots = [[] for i in range(2)]
        e = 1e-6
        e = float(input(f"Введите e или нажмите Enter для стандартного {e=}: ") or e)
        error = float('+inf')
        while error > e:
            graphic = [a]
            #Вычисление s и t
            s = []
            t = []
            s.append(-(alpha1 / 2 - alpha2 / h) / (alpha1 / 2 + alpha2 / h))
            t.append(alpha / (alpha1 / 2 + alpha2 / h))
            b = [0] * (n + 2)
            b[0] = alpha
            for i in range(1, n + 1):
                x = a - h / 2 + h * i
                graphic.append(x)
                b[i] = f(x)
                p_val = p(x)
                q_val = q(x)
                r_val = r(x)
                if mode == 1:
                    A = (-p_val / (h ** 2)) - q_val / (2 * h)
                    B = -(2 * p_val / (h ** 2)) - r_val
                    C = (-p_val / (h ** 2)) + q_val / (2 * h)
                else:
                    A = -p(a - h / 2 + h * (i - 1 / 2)) / h ** 2 - q_val / (2 * h)
                    B = -p(a - h / 2 + h * (i + 1 / 2)) / h ** 2 - p(a - h / 2 + h * (i - 1 / 2)) / h**2 - r_val
                    C = -p(a - h / 2 + h * (i + 1 / 2)) / h ** 2 + q_val / (2 * h)
                s.append(C / (B - A * s[i - 1]))
                t.append((A * t[i - 1] - b[i]) / (B - A * s[i - 1]))
            graphic.append(1)
            s.append(0)
            t.append(((betta1 / 2 - betta2 / h) * t[n] - betta) / (-(betta1 / 2 + betta2 / h) -(betta1 / 2 - betta2 / h) * s[n]))
            #Вычисление y на основе посчитанных s и t
            y = [0] * (n + 2)
            y[n + 1] = t[n + 1]
            for i in range(n, -1, -1):
                y[i] = s[i] * y[i + 1] + t[i]
            if len(solves) == 2: #После того, как просчитали два решение вычисляем ошибку
                er = 0
                for i in range(0, len(solves[1])):
                    if i % 2 == 0:
                        er += (((0.75 * solves[0][i // 2] + 0.25 * solves[0][i // 2 + 1]) - solves[1][i]) / 3) ** 2
                    else:
                        er += (((0.25 * solves[0][i // 2] + 0.75 * solves[0][i // 2 + 1]) - solves[1][i]) / 3) ** 2
                er /= len(solves[1])
                er = er ** (1 / 2)
                error = er
                dots[0].append(error)
                dots[1].append(n)

            if len(solves) == 2:
                solves[0], solves[1] = solves[1], list.copy(y)
            else:
                solves.append(list.copy(y))

            n *= 2
            h /= 2
        print(f"Итоговая погрешность: {dots[0][-1]}")
        plt.plot(graphic, solves[-1])
        plt.show()
        plt.plot(dots[1], dots[0])
        plt.show()
        # print(solves[-1])


if __name__ == '__main__':
    main()
