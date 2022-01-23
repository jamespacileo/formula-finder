from collections import defaultdict
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


def sympy_formula_to_tree(formula, tree=None, index=0, depth=0):
    if not tree:
        tree = [0] * 15
    name = formula.func.__name__
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
        is_divide = (
            second_arg.func.__name__ == "Pow"
            and second_arg.args[1].func.__name__ == "NegativeOne"
        )
        if is_divide:
            tree[index] = NODE_KEY_TO_INDEX["divide"]
            second_arg = second_arg.args[0]
        elif first_arg.func.__name__ == "NegativeOne":
            tree[index] = NODE_KEY_TO_INDEX["neg"]
            first_arg = second_arg
            second_arg = None
        else:
            tree[index] = NODE_KEY_TO_INDEX["multiply"]
        sympy_formula_to_tree(first_arg, tree, index * 2 + 1, depth + 1)
        if not second_arg:
            pass
        elif other_args:
            if is_divide:
                sympy_formula_to_tree(
                    Mul(second_arg, Pow(Mul(*other_args), Integer(-1))),
                    tree,
                    index * 2 + 2,
                    depth + 1,
                )
            else:
                sympy_formula_to_tree(
                    Mul(second_arg, *other_args), tree, index * 2 + 2, depth + 1
                )
        else:
            sympy_formula_to_tree(second_arg, tree, index * 2 + 2, depth + 1)
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
    elif name == "cos":
        tree[index] = NODE_KEY_TO_INDEX["cos"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "tan":
        tree[index] = NODE_KEY_TO_INDEX["tan"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "acos":
        tree[index] = NODE_KEY_TO_INDEX["acos"]
        sympy_formula_to_tree(formula.args[0], tree, index * 2 + 1, depth + 1)
    elif name == "atan":
        tree[index] = NODE_KEY_TO_INDEX["atan"]
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
    else:
        print(name)
        pass
    # for arg in formula.args:
    #     if arg.is_Number:
    #         tree.append(arg)
    #     else:
    #         sympy_formula_to_tree(arg, tree)

    return tree


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
