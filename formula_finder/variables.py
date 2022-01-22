import pygad
import numpy as np
import math
import random
from astropy import constants as c
from astropy import units as u
from typing import List

# list of functions for add, divide, multiply, subtract, mod, exponent, log, log10, sin, cos, tan, asin, acos, atan, sqrt, abs, and neg

SYMBOL_NODES = {
    "x": "x",
    "a": "a",
    "b": "b",
    "c": "c",
}
SYMBOL_NODES_LIST = ["x", "a", "b", "c"]
SYMBOL_INDICIES = list(range(len(SYMBOL_NODES_LIST)))
SYMBOL_KEYS = list(SYMBOL_NODES.keys())

MATH_FUNCTION_NODES = {
    "add": lambda x, y: x + y,
    "divide": lambda x, y: x / y,
    "multiply": lambda x, y: x * y,
    "subtract": lambda x, y: x - y,
    "mod": lambda x, y: x % y,
    "exponent": lambda x, y: x ** y,
    "log": lambda x, y: np.log(x),
    "log10": lambda x, y: np.log10(x),
    "sin": lambda x, y: np.sin(x),
    "cos": lambda x, y: np.cos(x),
    "tan": lambda x, y: np.tan(x),
    "asin": lambda x, y: np.arcsin(x),
    "acos": lambda x, y: np.arccos(x),
    "atan": lambda x, y: np.arctan(x),
    "sqrt": lambda x, y: np.sqrt(x),
    "abs": lambda x, y: np.abs(x),
    "neg": lambda x, y: -x,
}
MATH_FUNCTION_NODES_LIST = list(MATH_FUNCTION_NODES.keys())
MATH_FUNCTION_INDICIES = [
    i + len(SYMBOL_INDICIES) for i in range(len(MATH_FUNCTION_NODES_LIST))
]
MATH_FUNCTION_KEYS = list(MATH_FUNCTION_NODES.keys())

# list of mathematical constants

MATH_CONSTANT_NODES = {
    "pi": math.pi,
    "e": math.e,
}
MATH_CONSTANT_NODES_LIST = list(MATH_CONSTANT_NODES.keys())
MATH_CONSTANT_INDICIES = [
    i + len(SYMBOL_INDICIES) + len(MATH_FUNCTION_INDICIES)
    for i in range(len(MATH_CONSTANT_NODES_LIST))
]
MATH_CONSTANT_KEYS = list(MATH_CONSTANT_NODES.keys())

# list of astropy constants

ASTRO_CONSTANT_NODES = {
    "G": c.G.value,
    "c": c.c.value,
    "M_sun": c.M_sun.value,
    "N_A": c.N_A.value,
    "R": c.R.value,
    "h": c.h.value,
}
ASTRO_CONSTANT_NODES_LIST = list(ASTRO_CONSTANT_NODES.keys())
ASTRO_CONSTANT_INDICIES = [
    i + len(SYMBOL_INDICIES) + len(MATH_FUNCTION_INDICIES) + len(MATH_CONSTANT_INDICIES)
    for i in range(len(ASTRO_CONSTANT_NODES_LIST))
]
ASTRO_CONSTANT_KEYS = list(ASTRO_CONSTANT_NODES.keys())

CUSTOM_VARIABLE_NODES = {}  # {"m1": "m1", "m2": "m2"}
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


def update_indicies():
    global ALL_INDICIES
    global VARIABLE_INDICIES
    global CUSTOM_VARIABLES_INDICIES
    global CUSTOM_VARIABLES_KEYS
    global SYMBOL_INDICIES
    global SYMBOL_KEYS

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


def add_custom_variables(variables: List[str]):
    global CUSTOM_VARIABLE_NODES
    for variable in variables:
        CUSTOM_VARIABLE_NODES[variable] = variable
    update_indicies()


def get_node_from_index(index):
    """
    Returns a node from a given index
    """
    if index < len(SYMBOL_NODES_LIST):
        return "symbol", SYMBOL_NODES_LIST[index]
    index -= len(SYMBOL_NODES_LIST)
    if index < len(MATH_FUNCTION_NODES):
        return "math_func", MATH_FUNCTION_NODES[list(MATH_FUNCTION_NODES.keys())[index]]
    index -= len(MATH_FUNCTION_NODES)
    if index < len(MATH_CONSTANT_NODES):
        return (
            "math_const",
            MATH_CONSTANT_NODES[list(MATH_CONSTANT_NODES.keys())[index]],
        )
    index -= len(MATH_CONSTANT_NODES)
    if index < len(ASTRO_CONSTANT_NODES):
        return (
            "astro_const",
            ASTRO_CONSTANT_NODES[list(ASTRO_CONSTANT_NODES.keys())[index]],
        )
    index -= len(ASTRO_CONSTANT_NODES)
    if index < len(CUSTOM_VARIABLE_NODES):
        return (
            "custom_var",
            CUSTOM_VARIABLE_NODES[list(CUSTOM_VARIABLE_NODES.keys())[index]],
        )
    else:
        raise Exception("Index out of range")


# array to binary tree of nodes
def convert_array_to_binary_tree(array: List, x, a, b, c, symbols: List, array_index=0):
    # print("array_index", array_index)
    index = array[array_index]

    node_type, node = get_node_from_index(index)
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
            array, x, a, b, c, symbols, array_index=2 * array_index + 1
        )
        node2 = convert_array_to_binary_tree(
            array, x, a, b, c, symbols, array_index=2 * array_index + 2
        )
        return node(node1, node2)
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


def add_x_and_custom_variables_to_ending_indicies(tree):
    # make sure at least 1 ending indicies is x
    # print("tree", tree)
    ending_indicies = get_ending_indicies(tree)
    if len(ending_indicies) < len(CUSTOM_VARIABLES_INDICIES) + 1:
        raise NotEnoughEndingIndiciesException("Not enough ending indicies")
    # print("ending_indicies", ending_indicies)
    used_ending_indicies = []
    available_ending_indicies = [
        x for x in ending_indicies if x not in CUSTOM_VARIABLES_INDICIES + [0]
    ]
    variables_to_apply = [
        x for x in CUSTOM_VARIABLES_INDICIES + [0] if x not in ending_indicies
    ]
    for variable in variables_to_apply:
        if variable not in ending_indicies:
            random_index = random.choice(available_ending_indicies)
            tree[random_index] = variable
            available_ending_indicies.remove(random_index)

    return tree


if __name__ == "__main__":
    print(CUSTOM_VARIABLES_INDICIES)
    convert_array_to_binary_tree(np.array([4, 1, 3]), 1, 2, 3, 4, [29, 30])
    add_x_to_ending_indicies(np.array([4, 1, 3]))
    add_x_and_custom_variables_to_ending_indicies(
        np.array([5, 6, 2, 25, 0, 23, 11, 27, 0, 3, 26, 21, 2, 26, 2])
    )
