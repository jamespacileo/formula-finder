# import random
import numpy as np
from formula_finder.variables import (
    MATH_FUNCTION_1_PARAM_INDICIES,
    MATH_FUNCTION_2_PARAM_INDICIES,
    VARIABLE_INDICIES,
    ALL_INDICIES,
    REQUIRED_VARIABLE_INDICIES,
    NON_REQUIRED_VARIABLE_INDICIES,
    MATH_FUNCTION_TRIGONOMETRIC_INDICIES,
)
from formula_finder.binary_tree import (
    append_tree_to_tree,
    cut_tree_at_node_index,
    insert_child_into_parent_tree,
    non_blank_node_indicies,
    number_of_nodes_for_tree,
    depth_of_tree,
    randomly_cut_tree_at_depth,
)

BLANK_INDEX = -1
TWO_PARAM_NODE_INDICIES = MATH_FUNCTION_2_PARAM_INDICIES
ONE_PARAM_NODE_INDICIES = MATH_FUNCTION_1_PARAM_INDICIES


def pick_action(depth: int, max_depth: int):
    """
    depth starts at 0
    pick a random choice between add-node-2-param, add-node-1-param, add-variable
    gradually increase weight of add-variable as depth increases
    gradually decrease weight of add-node-2-param as depth increase
    """
    diff_depth = max_depth - depth
    if max_depth == 0:
        return "add_variable"
    add_node_weights = np.linspace(1.0, 0.0, max_depth + 1) ** 2
    add_variable_weights = np.linspace(0.0, 1.0, max_depth + 1)
    param_num_balance = np.array([0.6, 0.4])
    add_2_param_weight, add_1_param_weight = param_num_balance * add_node_weights[depth]
    add_variable_weight = add_variable_weights[depth]
    weights = np.array([add_2_param_weight, add_1_param_weight, add_variable_weight])
    weights = weights / weights.sum()
    return np.random.choice(
        ["add_node_2_param", "add_node_1_param", "add_variable"],
        p=weights,
    )


def variable_tree_node_indicies(tree: list):
    return [index for index, value in enumerate(tree) if value in VARIABLE_INDICIES]


def generate_tree(
    max_depth: int = None,
    tree: list = None,
    index: int = 0,
    depth: int = 0,
    required_variables_left: list = None,
):
    # action = random.choice(["add_node", "add_leaf", "add_constant"])
    # number_of_nodes = number_of_nodes_for_tree
    if tree is None:
        num_genes = number_of_nodes_for_tree(max_depth)
        tree = np.full((num_genes,), BLANK_INDEX, dtype=int)

    # if not max_depth:
    #     max_depth = depth_of_tree(num_genes)

    action = pick_action(depth, max_depth)
    if action == "add_node_2_param":
        tree[index] = np.random.choice(TWO_PARAM_NODE_INDICIES)
        generate_tree(
            max_depth,
            tree,
            index * 2 + 1,
            depth + 1,
            required_variables_left,
        )
        generate_tree(
            max_depth,
            tree,
            index * 2 + 2,
            depth + 1,
            required_variables_left,
        )
    elif action == "add_node_1_param":
        tree[index] = np.random.choice(ONE_PARAM_NODE_INDICIES)
        generate_tree(
            max_depth,
            tree,
            index * 2 + 1,
            depth + 1,
            required_variables_left,
        )
    elif action == "add_variable":
        if required_variables_left:
            choice = np.random.choice(required_variables_left)
            required_variables_left.remove(choice)
            tree[index] = choice
        else:
            tree[index] = np.random.choice(NON_REQUIRED_VARIABLE_INDICIES)

    if depth == 0:
        # Add x if missing
        variable_indicies = variable_tree_node_indicies(tree)
        has_x = 0 in variable_indicies
        if not has_x:
            tree[np.random.choice(variable_indicies)] = 0

    return tree


def tree_crossover(tree1, tree2):
    max_depth = depth_of_tree(len(tree1))
    available_indicies = non_blank_node_indicies(tree1)
    chosen_index = np.random.choice(available_indicies)

    parent1, child1 = cut_tree_at_node_index(tree1, chosen_index)
    child_depth = depth_of_tree(len(child1))

    max_depth2 = depth_of_tree(len(tree2)) - child_depth
    num_items_at_depth2 = number_of_nodes_for_tree(max_depth2)
    max_index2 = num_items_at_depth2 - 1
    available_indicies2 = [
        index for index in non_blank_node_indicies(tree2) if index <= max_index2
    ]

    chosen_index2 = np.random.choice(available_indicies2)
    parent2, child2 = cut_tree_at_node_index(tree2, chosen_index2)
    return insert_child_into_parent_tree(child1, parent2, chosen_index2)


def generate_population(total_pop_size, gene_size):
    max_depth = depth_of_tree(gene_size)
    population = []
    for i in range(total_pop_size):
        population.append(generate_tree(max_depth))
    return population


def prune_chromosome(tree: list):
    available_indicies = non_blank_node_indicies(tree)
    max_index = np.max(available_indicies)
    max_depth = depth_of_tree(max_index)
    tree_depth = depth_of_tree(len(tree))
    probability_of_pruning = (max_depth / tree_depth) ** 2
    do_prune = np.random.choice(
        [True, False], p=[probability_of_pruning, 1 - probability_of_pruning]
    )
    if do_prune:
        chosen_index = np.random.choice(available_indicies[1:])
        # print("Prune: Yes", chosen_index)
        tree, _ = cut_tree_at_node_index(tree, chosen_index)
        tree[chosen_index] = np.random.choice(VARIABLE_INDICIES)
        return tree
    return tree


def scrable_variables(tree: list):
    new_tree = tree.copy()
    variable_indicies = variable_tree_node_indicies(tree)
    values = new_tree[variable_indicies]
    scrabled = np.random.shuffle(values)
    new_tree[variable_indicies] = values
    return new_tree


def mutate_tree_node(tree: list):
    available_indicies = non_blank_node_indicies(tree)
    chosen_index = np.random.choice(available_indicies)
    tree_depth = depth_of_tree(len(tree))
    max_depth = tree_depth - depth_of_tree(chosen_index + 1)
    chosen_depth = np.random.choice(range(min(max_depth + 1, 2)))
    mutated_tree = generate_tree(max_depth=chosen_depth)
    return insert_child_into_parent_tree(mutated_tree, tree, chosen_index)


def replace_random_variable_with_x(tree: list):
    variable_indicies = variable_tree_node_indicies(tree)
    chosen_index = np.random.choice(variable_indicies)
    tree = tree.copy()
    tree[chosen_index] = 0
    return tree


def formula_mutation(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        new_chromosome = prune_chromosome(offspring[chromosome_idx])
        # new_chromosome = mutate_tree_node(new_chromosome)
        # if np.random.choice([True, False]):
        #     new_chromosome = replace_random_variable_with_x(new_chromosome)
        # if np.random.choice([True, False]):
        #     new_chromosome = scrable_variables(new_chromosome)
        offspring[chromosome_idx] = new_chromosome
    return offspring


def formula_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    # print("starting offspring", offspring, parents)
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        chromosome = tree_crossover(parent1, parent2)
        offspring.append(chromosome)

        idx += 1
    return np.array(offspring)


if __name__ == "__main__":
    for _ in range(1000):
        # print(pick_action(1, 3))
        tree1 = generate_tree(max_depth=3)
        # print(
        #     tree1,
        #     mutate_tree_node(tree1),
        # )
        # tree2 = generate_tree(max_depth=3)
        # parent, child = randomly_cut_tree_at_depth(tree, 1)
        # print(tree, parent, child)
        # print(generate_tree(3))
        # print(tree1, tree2, tree_crossover(tree1, tree2))
