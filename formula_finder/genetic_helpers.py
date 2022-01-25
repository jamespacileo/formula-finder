import math
import random
from scipy.optimize import curve_fit
import numpy as np
from formula_finder.binary_tree import (
    non_blank_node_indicies,
    number_of_leaf_nodes_for_tree,
)
from formula_finder.common_formulas import LIST_OF_COMMON_FORMULAS_AS_TREE
from formula_finder.generation import variable_tree_node_indicies
from formula_finder.simplify import (
    FormulaIsNotAllowedToBeANumber,
    simplify_formula_and_add_padding,
)
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


def generate_population(total_population_size, gene_size, use_common_formulas=True):
    # Generates a population of trees, where at least 1 leaf is x, and all leaves are either a symbol, math constant or astro constant
    population = [] + LIST_OF_COMMON_FORMULAS_AS_TREE
    tree_length = gene_size
    tree_leaf_count = number_of_leaf_nodes_for_tree(gene_size)
    while len(population) < total_population_size:
        # Generate a random tree where the nodes can be any index in the list of all_indicies
        # and the leaves must be variable indicies
        tree_nodes = np.random.choice(
            ALL_INDICIES, size=tree_length - tree_leaf_count, replace=True
        )
        tree_leaves = np.random.choice(
            VARIABLE_INDICIES, size=tree_leaf_count, replace=True
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

    return population


def fitness_function_factory(
    comparison_func, xdata, dim1_data=None, dim2_data=None, dimension_names=["x"]
):
    if dim1_data is None:
        dim1_data = np.zeros(xdata.shape)
    if dim2_data is None:
        dim2_data = np.zeros(xdata.shape)

    data = np.array([xdata, dim1_data, dim2_data])

    def fitness_function(solution_array, solution_idx):
        has_x = len([i for i in solution_array if i == 0]) > 0

        # does solution array contain 0?
        if not has_x:
            return 0

        if solution_array[0] in VARIABLE_INDICIES:
            return 0

        def func_to_fit(data, a, b, c):
            x, dim2, dim3 = data[0], data[1], data[2]
            return convert_array_to_binary_tree(
                solution_array, x, a, b, c, [dim2, dim3]
            )

        try:
            ptot, pcov = curve_fit(
                func_to_fit,
                data,
                comparison_func(data),
            )
            ydata = func_to_fit(data, *ptot)

            # if not ensure_tree_endings_end_with_constant(solution_array):
            #     return 0

            # node_endings = get_ending_indicies(solution_array)
            # node_endings = variable_tree_node_indicies(solution_array)
            # node_ending_count = len(node_endings)

            non_blanks = non_blank_node_indicies(solution_array)

            diff = np.sum(np.abs(ydata - comparison_func(data)) ** 2) * len(non_blanks)
            if diff == 0:
                return np.Inf
            fitness = 1 / diff

            # return np.sqrt(np.diag(pcov))
        # except RuntimeError:
        #     fitness = 0
        except ZeroDivisionError:
            # TODO: Double check that this is ok to have
            fitness = 0
        # except TypeError:
        #     fitness = 0
        # except Exception:
        #     fitness = 0
        # print("fitness", fitness)
        if np.isnan(fitness):
            fitness = 0
        return fitness

    return fitness_function


def custom_mutation(offspring, ga_instance):
    # print("mutation")
    for chromosome_idx in range(offspring.shape[0]):
        for retry_count in range(10):
            tree = offspring[chromosome_idx]
            try:
                tree = simplify_formula_and_add_padding(
                    tree, fail_formulas_that_are_numbers=True
                )
            except (
                FormulaIsNotAllowedToBeANumber,
                IndexError,
                ZeroDivisionError,
            ) as err:
                tree = random.choice(LIST_OF_COMMON_FORMULAS_AS_TREE)
            except TypeError as err:
                # TODO: Handle specific cases
                tree = random.choice(LIST_OF_COMMON_FORMULAS_AS_TREE)
            # try:
            #     tree = simplify_formula_and_add_padding(tree)
            # except:
            #     pass
            number_of_mutations = np.random.randint(1, 2)
            chromosome = offspring[chromosome_idx]
            for _ in range(number_of_mutations):
                tree_leaf_count = number_of_leaf_nodes_for_tree(len(tree))
                node_idx = random.randint(0, len(tree) - 1)

                if node_idx < len(tree) - tree_leaf_count:
                    # change a leaf node
                    chromosome[node_idx] = random.choice(ALL_INDICIES)
                else:
                    # change a node
                    chromosome[node_idx] = random.choice(VARIABLE_INDICIES)

            try:
                chromosome = add_x_and_custom_variables_to_ending_indicies(chromosome)
            except NotEnoughEndingIndiciesException as e:
                # TODO: Better handling of cases where the tree is invalid
                if retry_count > 5:
                    raise Exception("Too many tries")

            offspring[chromosome_idx] = chromosome

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


if __name__ == "__main__":
    from astropy import constants as c

    def default_comparison_func(data):
        x = data[0]
        m1 = data[1]
        m2 = data[2]
        return c.G.value * (m1 * m2) / np.power(x, 2)

    defaultCustomParametersData = {
        "m1": np.linspace(0.8, 1.2, 10) * c.M_sun.value,
        "m2": np.linspace(0.8, 1.2, 10) * c.M_earth.value,
    }
    xdata = np.linspace(0.8, 1.3, 10) * 14e9

    fitness_func = fitness_function_factory(
        default_comparison_func, xdata, None, None, dimension_names=["r"]
    )

    fitness_func([4, 6, 4, 2, 26, 24, 0, 22, 3, 26, 21, 26, 23, 0, 25], 1)
