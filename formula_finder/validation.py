from formula_finder.binary_tree import number_of_leaf_nodes_for_tree
import numpy as np

from formula_finder.variables import (
    ASTRO_CONSTANT_INDICIES,
    MATH_CONSTANT_INDICIES,
    SYMBOL_INDICIES,
    VARIABLE_INDICIES,
    get_ending_indicies,
)


def ensure_tree_leaf_nodes_end_with_constant(tree):
    tree_leaf_count = number_of_leaf_nodes_for_tree(len(tree))
    tree_leaves = np.array(tree[-tree_leaf_count:])
    # print(tree_leaves)
    # is at least one of the leaves x?
    if not tree_leaves.any(where=lambda x: x == 0):
        return 0
    # list of all symbol, math constant or astropy constant node indicies
    indicies = [*SYMBOL_INDICIES, *MATH_CONSTANT_INDICIES, *ASTRO_CONSTANT_INDICIES]
    # are all leaves either a symbol, math constant or astro constant?
    if any(x not in VARIABLE_INDICIES for x in tree_leaves):
        return 0
    return 1


def ensure_tree_endings_end_with_constant(tree):
    tree = np.array(tree)
    ending_indicies = get_ending_indicies(tree.tolist())
    ending_nodes = np.array(tree[ending_indicies])
    # if not ending_nodes.any(where=lambda x: x == 0):
    #     raise Exception("tree endings don't contain x ")
    # are all leaves either a symbol, math constant or astro constant?
    if any(x not in VARIABLE_INDICIES for x in ending_nodes):
        raise Exception("tree endings don't contain a variable")
    return 1
