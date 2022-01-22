import math
from formula_finder.variables import (
    ASTRO_CONSTANT_NODES,
    CUSTOM_VARIABLE_NODES,
    MATH_CONSTANT_NODES,
    MATH_FUNCTION_KEYS,
    MATH_FUNCTION_NODES,
    MATH_FUNCTION_NODES_LIST,
    get_node_from_index,
)
from typing import List


def number_of_nodes_for_tree(depth):
    return 2 ** (depth + 1) - 1


def depth_of_tree(number_of_nodes):
    return math.floor(math.log(number_of_nodes + 1, 2))


def number_of_leaf_nodes_in_binary_tree(depth):
    return (2 ** depth) - 1


def number_of_leaf_nodes_for_tree(number_of_nodes):
    return number_of_nodes - (2 ** (depth_of_tree(number_of_nodes) - 1)) + 1


number_of_leaf_nodes_for_tree(number_of_nodes_for_tree(3))


def convert_array_nodes_to_keys(tree: List):
    """
    Converts tree nodes to their respective key constant, math_function, math_const and astro_const strings
    """
    # print(tree)

    alpha_tree = []
    for index, node_index in enumerate(tree):
        # print("node index", index, node_index)
        node_type, node = get_node_from_index(node_index)
        # print(node_index, node_type, node)
        if node_type == "symbol":
            alpha_tree.append(node)
        if node_type == "math_func":
            # print(list(math_function_nodes.keys()), node_index, node_index - len(symbol_indicies))
            index = list(MATH_FUNCTION_NODES.values()).index(node)
            key = list(MATH_FUNCTION_NODES.keys())[index]
            alpha_tree.append(key)
        if node_type in ["math_const"]:
            index = list(MATH_CONSTANT_NODES.values()).index(node)
            key = list(MATH_CONSTANT_NODES.keys())[index]
            alpha_tree.append(key)
        if node_type in ["astro_const"]:
            index = list(ASTRO_CONSTANT_NODES.values()).index(node)
            key = list(ASTRO_CONSTANT_NODES.keys())[index]
            alpha_tree.append(key)
        if node_type in ["custom_var"]:
            index = list(CUSTOM_VARIABLE_NODES.values()).index(node)
            key = list(CUSTOM_VARIABLE_NODES.keys())[index]
            alpha_tree.append(key)
    return alpha_tree


if __name__ == "__main__":
    convert_array_nodes_to_keys([0, 13, 9, 1, 9, 1, 7, 5, 5, 6, 19, 1, 3, 4, 4])
