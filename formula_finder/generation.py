# import random
import numpy as np
from formula_finder.common_formulas import LIST_OF_COMMON_FORMULAS_AS_TREE
from formula_finder.variables import (
    MATH_FUNCTION_1_PARAM_INDICIES,
    MATH_FUNCTION_2_PARAM_INDICIES,
    REQUIRED_VARIABLE_INDICIES,
    VARIABLE_INDICIES,
    ALL_INDICIES,
    NON_REQUIRED_VARIABLE_INDICIES,
)
from formula_finder.binary_tree import (
    append_tree_to_tree,
    cut_tree_at_node_index,
    insert_child_into_parent_tree,
    list_of_tree_nodes_below_node,
    minimum_depth_for_n_leaf_nodes,
    non_blank_node_indicies,
    number_of_nodes_for_tree,
    depth_of_tree,
    randomly_cut_tree_at_depth,
)

BLANK_INDEX = -1
TWO_PARAM_NODE_INDICIES = MATH_FUNCTION_2_PARAM_INDICIES
ONE_PARAM_NODE_INDICIES = MATH_FUNCTION_1_PARAM_INDICIES


def pick_action(depth: int, max_depth: int, required_variables_left: list):
    """
    depth starts at 0
    pick a random choice between add-node-2-param, add-node-1-param, add-variable
    gradually increase weight of add-variable as depth increases
    gradually decrease weight of add-node-2-param as depth increase
    """
    diff_depth = max_depth - depth
    if max_depth == 0:
        return "add_variable"
    add_node_weights = np.linspace(1.0, 0.0, max_depth + 1)
    add_variable_weights = np.linspace(0.0, 1.0, max_depth + 1) ** (
        2 + len(required_variables_left)
    )
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
        if required_variables_left is None:
            raise Exception("required_variables_left must be provided")
        num_genes = number_of_nodes_for_tree(max_depth)
        tree = np.full((num_genes,), BLANK_INDEX, dtype=int)
        required_variables = required_variables_left.copy()
        required_variables_left = required_variables_left.copy()

    # if not max_depth:
    #     max_depth = depth_of_tree(num_genes)

    action = pick_action(depth, max_depth, required_variables_left)
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
        if required_variables_left is not None and len(required_variables_left) > 0:
            choice = np.random.choice(required_variables_left)
            required_variables_left.remove(choice)
            tree[index] = choice
        else:
            tree[index] = np.random.choice(NON_REQUIRED_VARIABLE_INDICIES)

    if depth == 0:
        # Add x if missing
        # REQUIRED_VARIABLE_INDICIES = get_required_variables_indicies()
        variable_indicies = [x for x in tree if x in VARIABLE_INDICIES]
        required_variables_included = [
            x for x in variable_indicies if x in required_variables
        ]
        if len(required_variables_included) < len(required_variables):
            return None
        # has_x = 0 in variable_indicies
        # if not has_x:
        #     tree[np.random.choice(variable_indicies)] = 0

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


def generate_population(
    total_pop_size, gene_size, add_common_formulas=True, num_dimensions=1
):
    # REQUIRED_VARIABLE_INDICIES = get_required_variables_indicies()
    max_depth = depth_of_tree(gene_size)
    population = []
    pops_left = total_pop_size
    while pops_left > 0:
        required_variables = REQUIRED_VARIABLE_INDICIES[:num_dimensions]
        pop = generate_tree(max_depth, required_variables_left=required_variables)
        if pop is not None:
            population.append(pop)
            pops_left -= 1
    if add_common_formulas:
        population.extend(LIST_OF_COMMON_FORMULAS_AS_TREE)
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


def mutate_tree_node(tree: list, required_variables: list):
    available_indicies = non_blank_node_indicies(tree)
    chosen_index = np.random.choice(available_indicies)
    tree_depth = depth_of_tree(len(tree))
    max_depth = tree_depth - depth_of_tree(chosen_index + 1)

    indicies_below_chosen_index = [
        tree[i] for i in list_of_tree_nodes_below_node(tree, chosen_index)
    ]
    required_variables_in_chosen_tree = np.unique(
        [i for i in indicies_below_chosen_index if i in required_variables]
    ).tolist()
    min_depth = minimum_depth_for_n_leaf_nodes(len(required_variables_in_chosen_tree))
    if min_depth > max_depth:
        raise Exception("Something wrong")
    if min_depth == max_depth:
        chosen_depth = max_depth
    else:
        chosen_depth = np.random.choice(range(min_depth, max_depth))

    mutated_tree = None
    while mutated_tree is None:
        mutated_tree = generate_tree(
            max_depth=chosen_depth,
            required_variables_left=required_variables_in_chosen_tree,
        )
    return insert_child_into_parent_tree(mutated_tree, tree, chosen_index)


def replace_random_variable_with_x(tree: list):
    variable_indicies = variable_tree_node_indicies(tree)
    chosen_index = np.random.choice(variable_indicies)
    tree = tree.copy()
    tree[chosen_index] = 0
    return tree


def replace_tree_head(tree: list):
    # extract a child head from tree at depth 1
    tree_depth = depth_of_tree(len(tree))
    max_depth = tree_depth - 1
    available_indicies = non_blank_node_indicies(tree)
    chosen_index = np.random.choice(available_indicies)
    child_tree, _ = cut_tree_at_node_index(tree, chosen_index)

    # create a new tree
    new_tree = generate_tree(max_depth=max_depth, required_variables_left=[])

    return insert_child_into_parent_tree(child_tree, new_tree, np.random.choice([1, 2]))

    # create a new tree an insert the child head into it
    # return the new tree


def formula_mutation_factory(num_dimensions: int):
    required_variables = REQUIRED_VARIABLE_INDICIES[:num_dimensions]

    def formula_mutation(offspring, ga_instance):
        for chromosome_idx in range(offspring.shape[0]):
            action = np.random.choice(
                [
                    "prune",
                    "mutate",
                    "scramble",
                    #  "sandwhich",
                    "replace_head",
                    "complete_replace",
                ]
            )
            new_chromosome = offspring[chromosome_idx].copy()
            if action == "prune":
                new_chromosome = prune_chromosome(new_chromosome)
            if action == "mutate":
                new_chromosome = mutate_tree_node(new_chromosome, required_variables)
            # if action == "scramble":
            #     new_chromosome = scrable_variables(new_chromosome)
            if action == "sandwhich":
                pass
            if action == "replace_head":
                new_chromosome = replace_random_variable_with_x(new_chromosome)
            if action == "complete_replace":
                while 1:
                    new_chromosome = generate_tree(
                        4, required_variables_left=required_variables
                    )
                    if new_chromosome is not None:
                        break
            # if np.random.choice([True, False]):
            #     new_chromosome = replace_random_variable_with_x(new_chromosome)
            # if np.random.choice([True, False]):
            #     new_chromosome = scrable_variables(new_chromosome)
            offspring[chromosome_idx] = new_chromosome
        return offspring

    return formula_mutation


def formula_crossover_factory(num_dimensions: int):
    required_variable_indicies = REQUIRED_VARIABLE_INDICIES[:num_dimensions]

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

    return formula_crossover


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
