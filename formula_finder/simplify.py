from collections import defaultdict
from fractions import Fraction
import numpy as np
from sympy.solvers import solve
from sympy import (
    Pow,
    Symbol,
    Add,
    Mul,
    Integer,
    Mod,
    log,
    acos,
    cos,
    sin,
    tan,
    acos,
    atan,
    asin,
    sqrt,
    Abs,
)

import sys
from formula_finder.algo_tree_helpers import convert_array_nodes_to_keys
from formula_finder.binary_tree import (
    pad_binary_tree_with_missing_nodes,
    print_binary_tree,
)

sys.path.append("/home/james/Projects/Astrophysics/formula_finder/")

from formula_finder.variables import NODE_KEY_TO_INDEX, convert_array_to_binary_tree


def represent_formula_in_sympy(formula: np.array):
    """
    Converts a formula from a binary tree to a sympy expression
    """
    # print(formula)
    return convert_array_to_binary_tree(
        formula,
        Symbol("x"),
        Symbol("a"),
        Symbol("b"),
        Symbol("c"),
        [Symbol("m1"), Symbol("m2")],
        use_sympy=True,
    )
    # return solve(expression)


def solve_formula(formula: np.array, x, a, b, c):
    """
    Converts a formula from a binary tree to a sympy expression
    """
    expression = represent_formula_in_sympy(formula)
    # print("expression", expression)
    return solve(expression, x, a, b, c)


class FormulaContainsAnUnknownFloat(Exception):
    pass


class FormulaIsNotAllowedToBeANumber(Exception):
    pass


def sympy_formula_to_tree(
    formula, tree=None, index=0, depth=0, fail_formulas_that_are_numbers=True
):
    if not tree:
        tree = [0] * 15

    is_number = isinstance(formula, (int, float, Fraction))
    if fail_formulas_that_are_numbers and depth == 0 and is_number:
        raise FormulaIsNotAllowedToBeANumber(formula.__str__())

    name = formula.func.__name__
    if (
        fail_formulas_that_are_numbers
        and depth == 0
        and name in ["Integer", "Float", "Zero", "Rational"]
    ):
        raise FormulaIsNotAllowedToBeANumber(formula.__str__())

    if name == "Add":
        first_arg, second_arg, *other_args = formula.args
        is_subtract = (
            second_arg.func.__name__ == "Mul"
            and second_arg.args[0].func.__name__ == "NegativeOne"
        )
        if is_subtract:
            tree[index] = NODE_KEY_TO_INDEX["subtract"]
            second_arg = second_arg.args[1]
        else:
            tree[index] = NODE_KEY_TO_INDEX["add"]
        sympy_formula_to_tree(first_arg, tree, index * 2 + 1, depth + 1)
        if other_args:
            if is_subtract:
                sympy_formula_to_tree(
                    Add(second_arg, Mul(Add(*other_args), Integer(-1))),
                    tree,
                    index * 2 + 2,
                    depth + 1,
                )
            else:
                sympy_formula_to_tree(
                    Add(second_arg, *other_args), tree, index * 2 + 2, depth + 1
                )

        else:
            sympy_formula_to_tree(second_arg, tree, index * 2 + 2, depth + 1)

    elif name == "Mul":
        first_arg, second_arg, *other_args = formula.args

        second_arg_is_divide = (
            second_arg.func.__name__ == "Pow" and second_arg.args[1] == -1
        )
        first_arg_is_divide = (
            first_arg.func.__name__ == "Pow" and first_arg.args[1] == -1
        )
        is_divide = first_arg_is_divide or second_arg_is_divide

        first_arg_is_negative = first_arg.func.__name__ == "NegativeOne"
        second_arg_is_negative = second_arg.func.__name__ == "NegativeOne"

        if first_arg_is_divide:
            tree[index] = NODE_KEY_TO_INDEX["divide"]
            divide_arg = first_arg.args[0]
            left_arg = Mul(second_arg, *other_args) if other_args else second_arg
            sympy_formula_to_tree(left_arg, tree, index * 2 + 1, depth + 1)
            sympy_formula_to_tree(divide_arg, tree, index * 2 + 2, depth + 1)
                    # second_arg = formula.args[1]
        elif second_arg_is_divide:
            tree[index] = NODE_KEY_TO_INDEX["divide"]
            divide_arg = second_arg.args[0]
            left_arg = Mul(first_arg, *other_args) if other_args else first_arg
            sympy_formula_to_tree(left_arg, tree, index * 2 + 1, depth + 1)
            sympy_formula_to_tree(divide_arg, tree, index * 2 + 2, depth + 1)
                    # second_arg = second_arg.args[]
        elif first_arg_is_negative:
            tree[index] = NODE_KEY_TO_INDEX["neg"]
            left_arg = second_arg
            second_arg = None
            sympy_formula_to_tree(left_arg, tree, index * 2 + 1, depth + 1)
        elif second_arg_is_negative:
            tree[index] = NODE_KEY_TO_INDEX["neg"]
            left_arg = first_arg
            second_arg = None
            sympy_formula_to_tree(left_arg, tree, index * 2 + 1, depth + 1)
            # sympy_formula_to_tree(divide_arg, tree, index * 2 + 2, depth + 1)
        else:
            tree[index] = NODE_KEY_TO_INDEX["multiply"]
            right_arg = Mul(second_arg, *other_args) if other_args else second_arg
            sympy_formula_to_tree(first_arg, tree, index * 2 + 1, depth + 1)

    elif name == "Pow":
        first_arg, second_arg = formula.args
        is_sqrt = second_arg.func.__name__ == "Half"
        if is_sqrt:
            tree[index] = NODE_KEY_TO_INDEX["sqrt"]
        else:
            tree[index] = NODE_KEY_TO_INDEX["power"]
        sympy_formula_to_tree(first_arg, tree, index * 2 + 1, depth + 1)
        if not is_sqrt:
            sympy_formula_to_tree(second_arg, tree, index * 2 + 2, depth + 1)

    elif name == "Symbol":
        symbol = formula.__str__()
        tree[index] = NODE_KEY_TO_INDEX[symbol]
    elif name == "log":
        tree[index] = NODE_KEY_TO_INDEX["log"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "sin":
        tree[index] = NODE_KEY_TO_INDEX["sin"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "sinh":
        tree[index] = NODE_KEY_TO_INDEX["sinh"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "cos":
        tree[index] = NODE_KEY_TO_INDEX["cos"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "cosh":
        tree[index] = NODE_KEY_TO_INDEX["cosh"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "tan":
        tree[index] = NODE_KEY_TO_INDEX["tan"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "acos":
        tree[index] = NODE_KEY_TO_INDEX["acos"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "cot":
        tree[index] = NODE_KEY_TO_INDEX["cot"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "atan":
        tree[index] = NODE_KEY_TO_INDEX["atan"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "atanh":
        tree[index] = NODE_KEY_TO_INDEX["atanh"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "asin":
        tree[index] = NODE_KEY_TO_INDEX["asin"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "sqrt":
        tree[index] = NODE_KEY_TO_INDEX["sqrt"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "Abs":
        tree[index] = NODE_KEY_TO_INDEX["abs"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "exp":
        tree[index] = NODE_KEY_TO_INDEX["exp"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "Float" and formula.__str__() == "2.71828182845905":
        tree[index] = NODE_KEY_TO_INDEX["e"]
    elif name == "Mod":
        tree[index] = NODE_KEY_TO_INDEX["mod"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
        sympy_formula_to_tree(formula.args[1], tree, index * 2 + 2, depth + 1)
    elif name == "Float" and formula.__str__() == "3.14159265359":
        tree[index] = NODE_KEY_TO_INDEX["pi"]
    elif name == "One":
        tree[index] = NODE_KEY_TO_INDEX["one"]
    elif name == "NegativeOne":
        tree[index] = NODE_KEY_TO_INDEX["negative_one"]
    elif name == "Half":
        tree[index] = NODE_KEY_TO_INDEX["half"]
    elif name == "Rational" and formula == Fraction(3, 4):
        tree[index] = NODE_KEY_TO_INDEX["three_quarters"]
    elif name in ["ComplexInfinity", "Infinity", "NegativeInfinity"]:
        tree[index] = NODE_KEY_TO_INDEX["c"]
    elif name == "Zero":
        tree[index] = NODE_KEY_TO_INDEX["c"]
    elif name in ["Float", "Integer", "Rational"]:
        tree[index] = NODE_KEY_TO_INDEX["c"]
    elif name == "Pi":
        tree[index] = NODE_KEY_TO_INDEX["pi"]
    elif name == "Exp1":
        tree[index] = NODE_KEY_TO_INDEX["e"]
    elif name == "re":
        sympy_formula_to_tree(formula.args[0], tree, index, depth)
    elif name in ["ImaginaryUnit", "im"]:
        # TODO Add imginary units
        tree[index] = NODE_KEY_TO_INDEX["c"]
    elif name == "NaN":
        tree[index] = NODE_KEY_TO_INDEX["c"]
    elif name == "AccumulationBounds":
        tree[index] = NODE_KEY_TO_INDEX["c"]
    else:
        print(name)
    # for arg in formula.args:
    #     if arg.is_Number:
    #         tree.append(arg)
    #     else:
    #         sympy_formula_to_tree(arg, tree)

    return tree


def show_sympy_tree(formula, depth=0):
    row = []
    for arg in formula.args:
        row += show_sympy_tree(arg, depth)
    if depth == 0:
        print(formula.func)
    print(row)
    return formula.func


def simplify_formula_and_add_padding(tree, fail_formulas_that_are_numbers=True):
    formula = represent_formula_in_sympy(tree)
    # print("formula:", formula)
    tree = sympy_formula_to_tree(
        formula, fail_formulas_that_are_numbers=fail_formulas_that_are_numbers
    )
    return pad_binary_tree_with_missing_nodes(tree)


if __name__ == "__main__":
    from formula_finder.variables import (
        add_custom_variables,
        SYMBOL_INDICIES,
        NODE_KEY_TO_INDEX,
    )
    from sympy import Symbol, symbols, sqrt
    import math
    import numpy as np

    add_custom_variables(["m1", "m2"])
    # represent_formula_in_sympy(
    #     [ 4,  6,  4,  2, 20, 24,  0, 0,  3, 0, 2, 1, 0,  0, 0]
    # )
    x, a, b, c, d, m1, m2 = symbols("x a b c d m1 m2")
    # expr = a * m1 * m2 / (x * x)
    expr = sqrt(x)
    tree = sympy_formula_to_tree(expr)
    tree = pad_binary_tree_with_missing_nodes(tree)
    print_binary_tree(convert_array_nodes_to_keys(tree))
    print(tree)
