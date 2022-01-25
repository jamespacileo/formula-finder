from formula_finder.simplify import represent_formula_in_sympy, solve_formula
from scipy.optimize import curve_fit
from formula_finder.variables import convert_array_to_binary_tree
import numpy as np
from sympy import solve, symbols, Symbol
import math


def render_formula(ga_instance, data, comparison_func):

    sol = ga_instance.best_solution()[0]
    sym = represent_formula_in_sympy(sol)

    def func_to_fit(data, a, b, c):
        x, dim2, dim3 = data[0], data[1], data[2]
        return convert_array_to_binary_tree(sol, x, a, b, c, [dim2, dim3])

    ptot, pcov = curve_fit(
        func_to_fit,
        data,
        comparison_func(data),
    )

    func_to_fit(data, *ptot)

    a, b, c = symbols("a b c")
    return sym.subs({a: ptot[0], b: ptot[1], c: ptot[2]})
