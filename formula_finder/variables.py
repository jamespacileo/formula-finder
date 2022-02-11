import pygad
import numpy as np
import math
import random
from astropy import constants as c
from astropy import units as u
from typing import List
import sympy
from fractions import Fraction

from formula_finder.binary_tree import (
    depth_of_tree,
    number_of_leaf_nodes_in_binary_tree,
)

# list of functions for add, divide, multiply, subtract, mod, exponent, log, log10, sin, cos, tan, asin, acos, atan, sqrt, abs, and neg

SYMBOL_NODES = {
    "x": "x",
    "a": "a",
    "b": "b",
    # "c": "c",
}
SYMBOL_NODES_LIST = list(SYMBOL_NODES.keys())
SYMBOL_INDICIES = list(range(len(SYMBOL_NODES_LIST)))
SYMBOL_KEYS = list(SYMBOL_NODES.keys())

MATH_FUNCTION_NODES = {
    "add": lambda x, y: x + y,
    "divide": lambda x, y: x / y,
    "multiply": lambda x, y: x * y,
    "subtract": lambda x, y: x - y,
    "square": lambda x, y: np.power(x, 2),
    "cube": lambda x, y: np.power(x, 3),
    "neg": lambda x, y: np.negative(x),
    "exp": lambda x, y: np.exp(x),
    "log": lambda x, y: np.log(x),
    "sin": lambda x, y: np.sin(x),
    "cos": lambda x, y: np.cos(x),
    "cosh": lambda x, y: np.cosh(x),
    "sinh": lambda x, y: np.sinh(x),
    "tan": lambda x, y: np.tan(x),
    "cot": lambda x, y: np.arctan(x),
    "asin": lambda x, y: np.arcsin(x),
    "acos": lambda x, y: np.arccos(x),
    "atan": lambda x, y: np.arctan(x),
    "atanh": lambda x, y: np.arctanh(x),
    "sqrt": lambda x, y: np.sqrt(x),
    "abs": lambda x, y: np.abs(x),
}

MATH_FUNCTION_NODES_USING_SYMPY = {
    "add": lambda x, y: x + y,
    "divide": lambda x, y: x / y,
    "multiply": lambda x, y: x * y,
    "subtract": lambda x, y: x - y,
    "square": lambda x, y: x ** 2,
    "cube": lambda x, y: x ** 3,
    "neg": lambda x, y: -x,
    "exp": lambda x, y: sympy.exp(x),
    "log": lambda x, y: sympy.log(x),
    "sin": lambda x, y: sympy.sin(x),
    "cos": lambda x, y: sympy.cos(x),
    "cosh": lambda x, y: sympy.cosh(x),
    "sinh": lambda x, y: sympy.sinh(x),
    "tan": lambda x, y: sympy.tan(x),
    "cot": lambda x, y: sympy.cot(x),
    "asin": lambda x, y: sympy.asin(x),
    "acos": lambda x, y: sympy.acos(x),
    "atan": lambda x, y: sympy.atan(x),
    "atanh": lambda x, y: sympy.atanh(x),
    "sqrt": lambda x, y: sympy.sqrt(x),
    "abs": lambda x, y: sympy.Abs(x),
}


MATH_FUNCTION_NODES_LIST = list(MATH_FUNCTION_NODES.keys())
MATH_FUNCTION_INDICIES = [
    i + len(SYMBOL_INDICIES) for i in range(len(MATH_FUNCTION_NODES_LIST))
]
MATH_FUNCTION_1_PARAM_INDICIES = MATH_FUNCTION_INDICIES[4:]
MATH_FUNCTION_2_PARAM_INDICIES = MATH_FUNCTION_INDICIES[:4]
MATH_FUNCTION_TRIGONOMETRIC_INDICIES = MATH_FUNCTION_INDICIES[10:-1]
MATH_FUNCTION_KEYS = list(MATH_FUNCTION_NODES.keys())

# list of mathematical constants

MATH_CONSTANT_NODES = {
    "pi": math.pi,
    "e": math.e,
    "one": 1,
    "negative_one": -1,
    "half": 0.5,
    "three_quarters": 3 / 4
    # "half": Fraction(1, 2),
    # "three_quarters": Fraction(3, 4),
}
MATH_CONSTANT_NODES_LIST = list(MATH_CONSTANT_NODES.keys())
MATH_CONSTANT_IGNORE_KEYS = ["one", "half", "three_quarters"]
MATH_CONSTANT_INDICIES = [
    i + len(SYMBOL_INDICIES) + len(MATH_FUNCTION_INDICIES)
    for i in range(len(MATH_CONSTANT_NODES_LIST))
]
MATH_CONSTANT_ALLOWED = MATH_CONSTANT_INDICIES[:2]
MATH_CONSTANT_KEYS = list(MATH_CONSTANT_NODES.keys())

# list of astropy constants

ASTRO_CONSTANT_NODES = {
    # "G": c.G.value,
    # "c": c.c.value,
    # "M_sun": c.M_sun.value,
    # "N_A": c.N_A.value,
    # "R": c.R.value,
    # "h": c.h.value,
}
ASTRO_CONSTANT_NODES_LIST = list(ASTRO_CONSTANT_NODES.keys())
ASTRO_CONSTANT_INDICIES = [
    i + len(SYMBOL_INDICIES) + len(MATH_FUNCTION_INDICIES) + len(MATH_CONSTANT_INDICIES)
    for i in range(len(ASTRO_CONSTANT_NODES_LIST))
]
ASTRO_CONSTANT_KEYS = list(ASTRO_CONSTANT_NODES.keys())

CUSTOM_VARIABLE_NODES = {"d2": "d2", "d3": "d3"}  # {"m1": "m1", "m2": "m2"}
CUSTOM_VARIABLES_INDICIES = [
    i
    + len(SYMBOL_INDICIES)
    + len(MATH_FUNCTION_INDICIES)
    + len(MATH_CONSTANT_INDICIES)
    + len(ASTRO_CONSTANT_INDICIES)
    for i in range(len(CUSTOM_VARIABLE_NODES))
]
CUSTOM_VARIABLES_KEYS = list(CUSTOM_VARIABLE_NODES.keys())


ALL_INDICIES = (
    SYMBOL_INDICIES
    + MATH_FUNCTION_INDICIES
    + MATH_CONSTANT_INDICIES
    + ASTRO_CONSTANT_INDICIES
    + CUSTOM_VARIABLES_INDICIES
)
VARIABLE_INDICIES = (
    SYMBOL_INDICIES
    + MATH_CONSTANT_INDICIES
    + ASTRO_CONSTANT_INDICIES
    + CUSTOM_VARIABLES_INDICIES
)

# LAST_INDEX = max(ASTRO_CONSTANT_INDICIES)

REQUIRED_VARIABLE_INDICIES = [0] + CUSTOM_VARIABLES_INDICIES
NON_REQUIRED_VARIABLE_INDICIES = SYMBOL_INDICIES[1:] + MATH_CONSTANT_ALLOWED

NODE_KEY_TO_INDEX = {}


# def get_required_variables_indicies():
#     global REQUIRED_VARIABLE_INDICIES
#     return REQUIRED_VARIABLE_INDICIES


def get_index_from_node_key(key):
    if key in SYMBOL_KEYS:
        return SYMBOL_INDICIES[SYMBOL_KEYS.index(key)]
    if key in MATH_FUNCTION_KEYS:
        return MATH_FUNCTION_INDICIES[MATH_FUNCTION_KEYS.index(key)]
    if key in MATH_CONSTANT_KEYS:
        return MATH_CONSTANT_INDICIES[MATH_CONSTANT_KEYS.index(key)]
    if key in ASTRO_CONSTANT_KEYS:
        return ASTRO_CONSTANT_INDICIES[ASTRO_CONSTANT_KEYS.index(key)]
    if key in CUSTOM_VARIABLES_KEYS:
        return CUSTOM_VARIABLES_INDICIES[CUSTOM_VARIABLES_KEYS.index(key)]
    raise Exception(f"Key {key} not in any node list")


def build_node_key_to_index_mapping():
    global NODE_KEY_TO_INDEX
    for key in SYMBOL_KEYS:
        NODE_KEY_TO_INDEX[key] = get_index_from_node_key(key)
    for key in MATH_FUNCTION_KEYS:
        NODE_KEY_TO_INDEX[key] = get_index_from_node_key(key)
    for key in MATH_CONSTANT_KEYS:
        NODE_KEY_TO_INDEX[key] = get_index_from_node_key(key)
    for key in ASTRO_CONSTANT_KEYS:
        NODE_KEY_TO_INDEX[key] = get_index_from_node_key(key)
    for key in CUSTOM_VARIABLES_KEYS:
        NODE_KEY_TO_INDEX[key] = get_index_from_node_key(key)


build_node_key_to_index_mapping()


def update_indicies():
    global ALL_INDICIES
    global VARIABLE_INDICIES
    global CUSTOM_VARIABLES_INDICIES
    global CUSTOM_VARIABLES_KEYS
    global SYMBOL_INDICIES
    global SYMBOL_KEYS
    global REQUIRED_VARIABLE_INDICIES

    CUSTOM_VARIABLES_INDICIES = [
        i
        + len(SYMBOL_INDICIES)
        + len(MATH_FUNCTION_INDICIES)
        + len(MATH_CONSTANT_INDICIES)
        + len(ASTRO_CONSTANT_INDICIES)
        for i in range(len(CUSTOM_VARIABLE_NODES))
    ]
    CUSTOM_VARIABLES_KEYS = list(CUSTOM_VARIABLE_NODES.keys())
    ALL_INDICIES = (
        SYMBOL_INDICIES
        + MATH_FUNCTION_INDICIES
        + MATH_CONSTANT_INDICIES
        + ASTRO_CONSTANT_INDICIES
        + CUSTOM_VARIABLES_INDICIES
    )
    VARIABLE_INDICIES = (
        SYMBOL_INDICIES
        + MATH_CONSTANT_INDICIES
        + ASTRO_CONSTANT_INDICIES
        + CUSTOM_VARIABLES_INDICIES
    )

    REQUIRED_VARIABLE_INDICIES = [0] + CUSTOM_VARIABLES_INDICIES

    build_node_key_to_index_mapping()


def add_custom_variables(variables: List[str]):
    global CUSTOM_VARIABLE_NODES
    CUSTOM_VARIABLE_NODES = {variable: variable for variable in variables}
    update_indicies()


def get_node_from_index(node_index, use_sympy=False):
    """
    Returns a node from a given index
    """
    if node_index == -1:
        return "blank", " "
    if node_index < len(SYMBOL_NODES_LIST):
        return "symbol", SYMBOL_NODES_LIST[node_index]
    node_index -= len(SYMBOL_NODES_LIST)
    if node_index < len(MATH_FUNCTION_NODES):
        if use_sympy:
            return (
                "math_func",
                MATH_FUNCTION_NODES_USING_SYMPY[
                    list(MATH_FUNCTION_NODES_USING_SYMPY.keys())[node_index]
                ],
            )
        return (
            "math_func",
            MATH_FUNCTION_NODES[list(MATH_FUNCTION_NODES.keys())[node_index]],
        )
    node_index -= len(MATH_FUNCTION_NODES)
    if node_index < len(MATH_CONSTANT_NODES):
        return (
            "math_const",
            MATH_CONSTANT_NODES[list(MATH_CONSTANT_NODES.keys())[node_index]],
        )
    node_index -= len(MATH_CONSTANT_NODES)
    if node_index < len(ASTRO_CONSTANT_NODES):
        return (
            "astro_const",
            ASTRO_CONSTANT_NODES[list(ASTRO_CONSTANT_NODES.keys())[node_index]],
        )
    node_index -= len(ASTRO_CONSTANT_NODES)
    if node_index < len(CUSTOM_VARIABLE_NODES):
        return (
            "custom_var",
            CUSTOM_VARIABLE_NODES[list(CUSTOM_VARIABLE_NODES.keys())[node_index]],
        )
    raise Exception(f"Index {node_index} out of range")


# array to binary tree of nodes
def convert_array_to_binary_tree(
    array: List, x, a, b, c, symbols: List, array_index=0, use_sympy=False
):
    # print("array_index", array_index)
    index = array[array_index]

    node_type, node = get_node_from_index(index, use_sympy)
    if node_type == "symbol":
        if index == 0:
            return x
        elif index == 1:
            return a
        elif index == 2:
            return b
        elif index == 3:
            return c
    if node_type == "math_func":
        is_two_param = index in MATH_FUNCTION_2_PARAM_INDICIES
        if 2 * array_index + 2 >= len(array):
            raise Exception(
                "there should be no nodes on leaf nodes {VARIABLE_INDICIES} {array_index}".format(
                    VARIABLE_INDICIES=VARIABLE_INDICIES, array_index=array_index
                )
            )
        # mid_way = math.floor(len(rest) / 2)
        # print(rest, mid_way)
        # print(len(array), array, array_index, 2*array_index+2, 2*array_index+2 >= len(array))

        node1 = convert_array_to_binary_tree(
            array,
            x,
            a,
            b,
            c,
            symbols,
            array_index=2 * array_index + 1,
            use_sympy=use_sympy,
        )
        if is_two_param:
            node2 = convert_array_to_binary_tree(
                array,
                x,
                a,
                b,
                c,
                symbols,
                array_index=2 * array_index + 2,
                use_sympy=use_sympy,
            )
            if node1 is None or node2 is None:
                print(node1, node2)
            try:
                return node(node1, node2)
            except Exception as err:
                print(err)
        return node(node1, None)
    if node_type in ["math_const", "astro_const"]:
        return node
    if node_type == "custom_var":
        return symbols[
            index
            - len(SYMBOL_INDICIES)
            - len(MATH_FUNCTION_INDICIES)
            - len(MATH_CONSTANT_INDICIES)
            - len(ASTRO_CONSTANT_INDICIES)
        ]


def get_ending_indicies(array: np.array, array_index=0):
    # print("array_index", array_index)
    index = array[array_index]
    ending_indicies = []

    node_type, node = get_node_from_index(index)
    if node_type == "symbol" and index in [0, 1, 2, 3]:
        return [array_index]
    if node_type == "math_func":
        if 2 * array_index + 2 >= len(array):
            raise Exception(
                "there should be no nodes on leaf nodes %s" % VARIABLE_INDICIES
            )
        # print(len(array), array, array_index, 2*array_index+2, 2*array_index+2 >= len(array))
        ending_indicies1 = get_ending_indicies(array, array_index=2 * array_index + 1)
        ending_indicies2 = get_ending_indicies(array, array_index=2 * array_index + 2)
        return np.concatenate([ending_indicies1, ending_indicies2])
        # if (ending_indicies1.size != 0) and (ending_indicies2.size != 0):
        #     return np.concatenate([ending_indicies1, ending_indicies2])
        # elif ending_indicies1.size != 0:
        #     return ending_indicies1
        # else:
        #     return ending_indicies2
    if node_type in ["math_const", "astro_const"]:
        return [array_index]
    if node_type == "custom_var":
        return [array_index]


def get_ending_indicies_that_are_not_leaf_nodes(array: np.array, array_index=0):
    ending_indicies = get_ending_indicies(array, array_index=array_index)
    leaf_count = number_of_leaf_nodes_in_binary_tree(depth_of_tree(len(array)))
    return [index for index in ending_indicies if index < len(array) - leaf_count]


def add_x_to_ending_indicies(tree):
    # make sure at least 1 ending indicies is x
    # print("tree", tree)
    ending_indicies = get_ending_indicies(tree)
    # print("ending_indicies", ending_indicies)
    if all(tree[index] != 0 for index in ending_indicies):
        random_index = random.choice(ending_indicies)
        tree[random_index] = 0
    return tree


class NotEnoughEndingIndiciesException(Exception):
    pass


def replace_ending_index_with_random_node(tree):
    ending_indicies = get_ending_indicies_that_are_not_leaf_nodes(tree)
    if len(ending_indicies) == 0:
        raise NotEnoughEndingIndiciesException(f"{tree}")
    random_index = random.choice(ending_indicies)
    tree[random_index] = random.choice(MATH_FUNCTION_2_PARAM_INDICIES)
    return tree


def add_x_and_custom_variables_to_ending_indicies(tree):
    # make sure at least 1 ending indicies is x
    # print("tree", tree)
    ending_indicies = get_ending_indicies(tree)

    # number of ending indicies that are not custom variables

    # nodes
    variables_to_apply = [
        x for x in CUSTOM_VARIABLES_INDICIES + [0] if x not in tree[ending_indicies]
    ]
    # tree indicies
    available_ending_indicies = [
        x for x in ending_indicies if tree[x] not in CUSTOM_VARIABLES_INDICIES + [0]
    ]
    # print("variables_to_apply", variables_to_apply)

    retries = 0
    # add ending nodes to fit more custom variables
    while len(available_ending_indicies) < len(variables_to_apply):
        print("")
        retries += 1
        if retries > 5:
            raise NotEnoughEndingIndiciesException("Not enough ending indicies")
        tree = replace_ending_index_with_random_node(tree)
        ending_indicies = get_ending_indicies(tree)
        variables_to_apply = [
            x for x in CUSTOM_VARIABLES_INDICIES + [0] if x not in tree[ending_indicies]
        ]
        available_ending_indicies = [
            x for x in ending_indicies if x not in CUSTOM_VARIABLES_INDICIES + [0]
        ]

    # print("ending_indicies", ending_indicies)
    # used_ending_indicies = []
    for variable in variables_to_apply:
        if variable not in tree[ending_indicies]:
            random_node_index = random.choice(available_ending_indicies)
            random_index = available_ending_indicies.index(random_node_index)
            tree[random_index] = variable
            available_ending_indicies.remove(random_node_index)

    return tree


if __name__ == "__main__":
    # print(CUSTOM_VARIABLES_INDICIES)
    # convert_array_to_binary_tree(np.array([4, 1, 3]), 1, 2, 3, 4, [29, 30])
    # add_x_to_ending_indicies(np.array([4, 1, 3]))
    add_custom_variables(["m1", "m2"])
    add_x_and_custom_variables_to_ending_indicies(
        np.array([22, 14, 23, 14, 24, 20, 13, 23, 0, 23, 24, 24, 24, 22, 21])
    )
