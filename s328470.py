# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5

def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])

def f2(x: np.ndarray) -> np.ndarray:
    return (abs((x[0] + x[0])) * ((np.exp(abs(x[0])) + abs(((abs((((x[0] - x[2]) + x[0]) / x[0])) + (x[0] + (((((x[0] + ((x[0] + (x[0] + (x[0] - x[2]))) * (abs((((x[0] + (x[0] + (((x[0] / x[0]) + (x[0] + ((x[0] - x[2]) + x[0]))) + (abs(x[2]) + x[0])))) * x[0]) + x[0])) + abs(x[0])))) + x[0]) + (x[0] + x[0])) + (x[0] + x[0])) * abs((x[0] + x[0]))))) + x[2]))) * (x[0] + ((x[0] + ((x[0] + x[0]) + x[0])) + x[0]))))

def f3(x: np.ndarray) -> np.ndarray:
    return (((((abs(x[0]) - (x[2] - (((abs(x[0]) - (x[2] - ((((abs(x[1]) - x[1]) - x[1]) - ((((x[1] - ((abs((x[0] * abs(x[0]))) - x[1]) + x[1])) - x[1]) - x[0]) - x[1])) - ((((x[0] - ((x[1] - x[1]) - x[1])) + x[1]) - x[1]) - x[1])))) - ((((((abs((x[1] + abs((x[1] * x[1])))) - x[1]) - x[1]) - ((abs(x[0]) - (abs(x[1]) - ((np.cos(((x[0] - x[0]) - x[0])) - x[1]) - x[2]))) - x[2])) - x[1]) - abs(x[0])) - x[1])) - ((x[1] - (abs(x[0]) - x[1])) - x[1])))) - x[1]) - (((x[1] - x[2]) - x[1]) - x[1])) - ((x[1] - x[1]) + x[2])) - ((((x[1] + abs((x[1] * x[1]))) - x[1]) - x[1]) * x[1]))

def f4(x: np.ndarray) -> np.ndarray:
    return ((np.cos(x[1]) + np.cos(x[1])) + (((np.cos(x[1]) + (np.cos(x[1]) + np.cos(x[1]))) + (np.cos(x[1]) + ((abs((abs((x[1] / x[1])) + abs((x[1] / x[1])))) + (np.cos(x[1]) / (np.cos(x[1]) + (np.cos(x[1]) + np.cos(x[1]))))) + (x[1] / x[1])))) + np.cos(x[1])))

def f5(x: np.ndarray) -> np.ndarray:
    return 0

def f6(x: np.ndarray) -> np.ndarray:
    return (x[1] + (((x[1] - x[0]) * np.sin(1)) * np.sin(1)))

def f7(x: np.ndarray) -> np.ndarray:
    return ((((x[0] + ((x[0] + ((x[0] * x[0]) * x[1])) * np.sin((x[1] * x[1])))) * (x[1] + ((x[0] + ((x[1] + x[0]) + x[1])) * np.sin((x[0] * x[0]))))) + (((x[1] + (np.sin((x[1] * (x[0] * x[0]))) + np.exp((x[0] * x[1])))) * x[1]) * x[0])) + (x[1] + (abs(x[1]) + np.exp((x[0] * x[1])))))

def f8(x: np.ndarray) -> np.ndarray:
    return ((((x[5] * abs(((x[5] + ((x[5] + x[5]) + x[5])) + x[5]))) + x[5]) + x[5]) * abs(((x[5] + (x[5] + x[5])) + (x[5] + ((x[5] + (((x[5] + ((x[5] + x[5]) + x[5])) / x[5]) + x[4])) + (((x[5] + (((x[5] + ((x[5] + (x[5] + ((x[2] + ((x[5] + x[5]) + x[5])) + x[5]))) - (x[2] + x[5]))) - (((x[5] + x[0]) + (((x[5] * abs(((x[5] + (((x[5] + x[5]) + x[5]) + x[5])) + x[5]))) + x[5]) + (x[5] + x[5]))) + x[5])) + x[5])) - (x[4] + (x[2] + x[5]))) + x[5]))))))