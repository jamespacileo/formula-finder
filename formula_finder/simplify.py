import numpy as np
from sympy.solvers import solve
from sympy import Symbol

import sys

sys.path.append("/home/james/Projects/Astrophysics/formula_finder/")

from formula_finder.variables import convert_array_to_binary_tree


def represent_formula_in_sympy(formula: np.array):
    """
    Converts a formula from a binary tree to a sympy expression
    """
    return convert_array_to_binary_tree(
        formula,
        Symbol("x"),
        Symbol("a"),
        Symbol("b"),
        Symbol("c"),
        [Symbol("m1"), Symbol("m2")],
    )
    # return solve(expression)


def solve_formula(formula: np.array, x, a, b, c):
    """
    Converts a formula from a binary tree to a sympy expression
    """
    expression = represent_formula_in_sympy(formula)
    # print("expression", expression)
    return solve(expression, x, a, b, c)


if __name__ == "__main__":
    from formula_finder.variables import add_custom_variables

    add_custom_variables(["m1", "m2"])
    represent_formula_in_sympy(
        [17, 7, 30, 29, 0, 6, 11, 28, 30, 29, 25, 23, 22, 26, 27]
    )
