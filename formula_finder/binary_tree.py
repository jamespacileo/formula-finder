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


def cut_binary_tree_at_node(tree: list, node_index: int):
    """
    Separates a binary tree into two trees at a given node
    """
    child_tree = []
    highest_index = node_index
    current_row = [node_index]
    while highest_index * 2 + 2 < len(tree):
        new_row = []
        for index in current_row:
            left_index = index * 2 + 1
            child_tree.append(left_index)
            new_row.append(index * 2 + 1)
            new_row.append(index * 2 + 2)
            highest_index = index * 2 + 2
        current_row = new_row


def random_cut_binary_tree(tree: list, depth: int):
    tree_depth = depth_of_tree(len(tree))
    node_indicies_at_depth = []


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


import numpy as np
from formula_finder.binary_tree import number_of_nodes_for_tree


def list_of_tree_nodes_below_node(tree: list, node_index: int) -> list:
    indicies_below_node = []
    current_index = node_index
    items_to_append = [node_index]
    while items_to_append:
        next_items_to_append = []
        for current_index in items_to_append:
            indicies_below_node.append(current_index)
            if current_index * 2 + 2 < len(tree):
                next_items_to_append.append(current_index * 2 + 1)
                next_items_to_append.append(current_index * 2 + 2)
        items_to_append = next_items_to_append
    return indicies_below_node


def non_blank_node_indicies(tree: list):
    return [i for i, node in enumerate(tree) if node != -1]


def node_indicies_at_depth(tree: list, depth: int):
    node_count = number_of_nodes_for_tree(depth)
    above_node_count = number_of_nodes_for_tree(depth - 1)
    num_nodes = node_count - above_node_count
    return np.linspace(
        above_node_count, above_node_count + num_nodes, num_nodes, dtype=int
    )


def cut_tree_at_node_index(tree: list, chosen_index: int):
    child_tree_indicies = list_of_tree_nodes_below_node(tree, chosen_index)
    parent_tree = tree.copy()
    parent_tree[child_tree_indicies] = -1
    child_tree = tree[child_tree_indicies]
    return parent_tree, child_tree


def randomly_cut_tree_at_depth(tree: list, depth: int):
    indicies = node_indicies_at_depth(tree, depth)
    chosen_index = np.random.choice(indicies)
    return cut_tree_at_node_index(tree, chosen_index)


def tree_crossover(tree_1: list, tree_2: list):
    """ """


def cut_binary_tree_at_node(tree: list, node_index: int):
    """
    Separates a binary tree into two trees at a given node
    """
    indicies_below_node = list_of_tree_nodes_below_node(tree, node_index)
    child_tree = tree[indicies_below_node]
    tree_without_child = []


def insert_child_into_parent_tree(
    child_tree: list, parent_tree: list, parent_index: int, child_index: int = 0
):
    new_tree = parent_tree.copy()
    new_tree[parent_index] = child_tree[child_index]
    if child_index * 2 + 2 < len(child_tree):
        new_tree = insert_child_into_parent_tree(
            child_tree, new_tree, parent_index * 2 + 1, child_index * 2 + 1
        )
        new_tree = insert_child_into_parent_tree(
            child_tree, new_tree, parent_index * 2 + 2, child_index * 2 + 2
        )
    return new_tree


def append_tree_to_tree(tree: list, tree_to_append: list, node_index: int):
    # n * + 1
    new_tree = tree.copy()
    tree_to_append = tree_to_append.copy()
    current_index = node_index

    items_to_append = [node_index]
    while items_to_append:
        next_items_to_append = []
        for current_index in items_to_append:
            new_tree[current_index] = current_index
            if current_index * 2 + 2 < len(new_tree):
                next_items_to_append.append(current_index * 2 + 1)
                next_items_to_append.append(current_index * 2 + 2)
        items_to_append = next_items_to_append
    return new_tree


# cut_binary_tree_at_node([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], 2)
# number_of_nodes_for_tree(3)


if __name__ == "__main__":
    list_of_tree_nodes_below_node([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2)
