import math
from collections import defaultdict


def number_of_nodes_for_tree(depth):
    return 2 ** (depth + 1) - 1


def depth_of_tree(number_of_nodes):
    return math.ceil(math.log(number_of_nodes + 1, 2))


def number_of_leaf_nodes_in_binary_tree(depth):
    return 2 ** (depth - 1)


def number_of_leaf_nodes_for_tree(number_of_nodes):
    return number_of_nodes - (2 ** (depth_of_tree(number_of_nodes) - 1)) + 1


# number_of_leaf_nodes_for_tree(number_of_nodes_for_tree(3))


def read_binary_tree(array, depth_tree=None, index=0, depth=0):
    """
    Converts a formula from a binary tree to a sympy expression
    """
    if not depth_tree:
        depth_tree = defaultdict(list)
    depth_tree[depth].append(array[index])
    if index * 2 + 1 < len(array):
        depth_tree = read_binary_tree(array, depth_tree, index * 2 + 1, depth + 1)
        depth_tree = read_binary_tree(array, depth_tree, index * 2 + 2, depth + 1)
    return depth_tree


PRINT_WIDTH = 80


def print_binary_tree(tree: list):
    tree_dict = read_binary_tree(tree)
    for key, value in tree_dict.items():
        line = "".join(
            node.center(int(PRINT_WIDTH / (2 ** key)), " ") for node in value
        )
        print(line)


def pad_binary_tree_with_missing_nodes(tree: list):
    depth = depth_of_tree(len(tree))
    number_of_nodes_needed = number_of_nodes_for_tree(depth)
    return tree + [0] * (number_of_nodes_needed - len(tree))
