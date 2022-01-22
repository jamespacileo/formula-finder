import random
from scipy.optimize import curve_fit
import numpy as np
from formula_finder.binary_tree import number_of_leaf_nodes_for_tree
from formula_finder.validation import ensure_tree_endings_end_with_constant

from formula_finder.variables import (
    ALL_INDICIES,
    CUSTOM_VARIABLES_INDICIES,
    VARIABLE_INDICIES,
    NotEnoughEndingIndiciesException,
    add_x_and_custom_variables_to_ending_indicies,
    add_x_to_ending_indicies,
    convert_array_to_binary_tree,
    get_ending_indicies,
)


def generate_population(total_population_size, gene_size):
    # Generates a population of trees, where at least 1 leaf is x, and all leaves are either a symbol, math constant or astro constant
    population = []
    tree_length = gene_size
    tree_leaf_count = number_of_leaf_nodes_for_tree(gene_size)
    while len(population) < total_population_size:
        # Generate a random tree where the nodes can be any index in the list of all_indicies
        # and the leaves must be variable indicies
        tree_nodes = np.random.choice(
            ALL_INDICIES, size=tree_length - tree_leaf_count, replace=False
        )
        tree_leaves = np.random.choice(
            VARIABLE_INDICIES, size=tree_leaf_count, replace=False
        )
        tree = np.concatenate((tree_nodes, tree_leaves))
        # Ensure that the tree has at least one leaf that is x
        # print("tree", convert_array_nodes_to_keys(tree.tolist()))
        # print("tree_nodes", convert_array_nodes_to_keys(tree_nodes.tolist()))
        # print("tree_leaves", convert_array_nodes_to_keys(tree_leaves.tolist()))
        try:
            tree = add_x_and_custom_variables_to_ending_indicies(tree)
        except NotEnoughEndingIndiciesException as e:
            continue
        # if not (ensure_tree_leaf_nodes_end_with_constant(tree) or ensure_tree_endings_end_with_constant(tree)):
        #     # Add the tree to the population
        #     raise Exception("Should not generate invalid population")
        population.append(tree)

        if len(population) == total_population_size:
            return population


def fitness_function_factory(comparison_func, xdata, customParameterData: dict = {}):
    def fitness_function(solution_array, solution_idx):
        variables = [
            random.choice(customParameterData[key]) for key in customParameterData
        ]

        def func_to_fit(x, dim2, dim3, a, b, c):
            return convert_array_to_binary_tree(solution_array, x, a, b, c, dim2, dim3)

        dim1_data = np.array([0])
        if len(customParameterData) > 0:
            dim1_data = customParameterData.values()[0]
        dim2_data = np.array([0])
        if len(customParameterData) > 1:
            dim2_data = customParameterData.values()[1]

        try:
            ptot, pcov = curve_fit(
                func_to_fit,
                xdata,
                dim1_data,
                dim2_data,
                comparison_func(xdata, dim1_data, dim2_data),
            )
            ydata = func_to_fit(xdata, *ptot)

            if not ensure_tree_endings_end_with_constant(solution_array):
                return 0

            node_endings = get_ending_indicies(solution_array)
            node_ending_count = len(node_endings)

            diff = np.sum((ydata - comparison_func(xdata, dim1_data, dim2_data)))
            if diff == 0:
                return 10e5 - node_ending_count
            fitness = 1 / diff

            # return np.sqrt(np.diag(pcov))
        except RuntimeError:
            fitness = 0
        # print("fitness", fitness)
        if np.isnan(fitness):
            fitness = 0
        return fitness

    return fitness_function


def custom_mutation(offspring, ga_instance):
    # print("mutation")
    for chromosome_idx in range(offspring.shape[0]):
        tree = offspring[chromosome_idx]
        tree_leaf_count = number_of_leaf_nodes_for_tree(len(tree))
        node_idx = random.randint(0, len(tree) - 1)
        if node_idx < len(tree) - tree_leaf_count:
            # change a leaf node
            offspring[chromosome_idx][node_idx] = random.choice(ALL_INDICIES)
        else:
            # change a node
            offspring[chromosome_idx][node_idx] = random.choice(VARIABLE_INDICIES)

        add_x_to_ending_indicies(offspring[chromosome_idx])

        if not ensure_tree_endings_end_with_constant(
            offspring[chromosome_idx].tolist()
        ):
            raise Exception("Invalid mutation")
    return offspring


def custom_crossover_func(parents, offspring_size, ga_instance):
    # print("crossover")
    offspring = []
    idx = 0
    # print("starting offspring", offspring, parents)
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_split_point = np.random.choice(range(offspring_size[0]))

        parent1[random_split_point:] = parent2[random_split_point:]

        offspring.append(parent1)

        idx += 1
    # print("ending offspring", offspring)

    if not ensure_tree_endings_end_with_constant(offspring[1]):
        raise Exception("Invalid crossover")
    return np.array(offspring)
