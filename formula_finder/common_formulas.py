import sympy
import numpy as np
from formula_finder.algo_tree_helpers import convert_array_nodes_to_keys
from formula_finder.simplify import sympy_formula_to_tree
from formula_finder.binary_tree import (
    number_of_nodes_for_tree,
    pad_binary_tree_with_missing_nodes,
    pad_tree_with_blank_nodes,
    print_binary_tree,
)

x, a, b, c, m1, m2 = sympy.symbols("x a b c m1 m2")

# list of commong math formulas

linear_formula = a * x

square_formula = a * (x * x)

# cube_formula = a * (x ** 3) + b * (x ** 2) + c * x

square_root_formula = a * sympy.sqrt(x)

absolute_formula = a * sympy.Abs(x)

reciprocal_formula = a / x

double_reciptocal_formula = a / (x * x)

logarithm_formula = a * sympy.log(x)

exponential_formula = a * sympy.exp(x)


# list of common trigonometric formulas

sine_formula = a * sympy.sin(x) + b

cosine_formula = a * sympy.cos(x) + b

tangent_formula = a * sympy.tan(x) + b

arc_sine_formula = a * sympy.asin(x) + b

arc_cosine_formula = a * sympy.acos(x) + b

arc_tangent_formula = a * sympy.atan(x) + b

LIST_OF_COMMON_FORMULAS = [
    linear_formula,
    square_formula,
    # cube_formula,
    square_root_formula,
    absolute_formula,
    reciprocal_formula,
    double_reciptocal_formula,
    logarithm_formula,
    exponential_formula,
    sine_formula,
    cosine_formula,
    tangent_formula,
    arc_sine_formula,
    arc_cosine_formula,
    arc_tangent_formula,
]

LIST_OF_COMMON_FORMULAS_AS_TREE = [
    sympy_formula_to_tree(formula) for formula in LIST_OF_COMMON_FORMULAS
]

LIST_OF_COMMON_FORMULAS_AS_TREE = [
    np.array(pad_tree_with_blank_nodes(formula))
    for formula in LIST_OF_COMMON_FORMULAS_AS_TREE
]

if __name__ == "__main__":
    # for each formula convert to tree and print
    num = number_of_nodes_for_tree(4)
    for formula in LIST_OF_COMMON_FORMULAS:
        # print(formula)
        tree = sympy_formula_to_tree(formula)
        tree = pad_binary_tree_with_missing_nodes(tree)
        print_binary_tree(convert_array_nodes_to_keys(tree))
