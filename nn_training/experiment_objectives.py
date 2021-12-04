import numpy as np
from math import sin, cos


def polynomial_func(x: float) -> float:
    return x ** 3 - 2*x**2

def sinusoidal(x: float) -> float:
    return sin(x) + 2 * cos(x / 3)

def xor(x) -> float:
    x, y = x
    return float(x != y)
