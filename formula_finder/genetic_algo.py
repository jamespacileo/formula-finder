import math
from typing import Callable, List
import pygad
import numpy as np
from tqdm import trange, tqdm
from formula_finder.algo_tree_helpers import convert_array_nodes_to_keys

from formula_finder.binary_tree import (
    number_of_nodes_for_tree,
)
from formula_finder.common_formulas import LIST_OF_COMMON_FORMULAS_AS_TREE
from formula_finder.generation import (
    formula_crossover_factory,
    formula_mutation_factory,
    generate_population,
)
from formula_finder.genetic_helpers import (
    custom_crossover_func,
    custom_mutation,
    fitness_function_factory,
)
from formula_finder.output import render_formula

mutation_percent_genes = 10


def run_genetic_algo(
    comparison_func: Callable,
    total_population_size: int = 1000,
    tree_depth: int = 4,
    num_generations: int = 100,
    xdata: List[int] = np.linspace(1, 10, 10),
    customParametersData: dict = None,
    dim1_data: List[int] = None,
    dim2_data: List[int] = None,
    dimension_names: List[str] = ["x"],
):

    num_genes = number_of_nodes_for_tree(tree_depth - 1)
    # population = np.concatenate(
    #     [generate_population(100, num_genes), LIST_OF_COMMON_FORMULAS_AS_TREE]
    # )
    num_dimensions = len(dimension_names)
    population = generate_population(100, num_genes, num_dimensions=num_dimensions)
    num_parents_mating = 50  # math.floor(total_population_size / 20)

    sol_per_pop = 50

    init_range_low = 0
    init_range_high = 20

    parent_selection_type = "sus"
    keep_parents = -1  # math.floor(total_population_size / 20)

    crossover_type = formula_crossover_factory(num_dimensions=num_dimensions)
    mutation_type = formula_mutation_factory(num_dimensions=num_dimensions)

    fitness_function = fitness_function_factory(
        comparison_func, xdata, dim1_data, dim2_data, num_dimensions=num_dimensions
    )

    t = tqdm(total=num_generations)

    def callback_gen(ga_instance):
        # print("Generation : ", ga_instance.generations_completed)
        best_solution = ga_instance.best_solution()
        # print("Fitness of the best solution :", best_solution[1])
        # print("Current best solution :", convert_array_nodes_to_keys(best_solution[0]))
        t.update(1)
        t.set_description_str(
            f"Fitness: {best_solution[1]}"  # {convert_array_nodes_to_keys(best_solution[0])}"
        )
        if ga_instance.best_solution()[1] == np.Inf:
            return "stop"

    # def func_generation(ga_instance):
    #     if ga_instance.best_solution()[1] == np.Inf:
    #         return "stop"

    ga_instance = pygad.GA(
        initial_population=population,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        gene_type=int,
        mutation_percent_genes=mutation_percent_genes,
        callback_generation=callback_gen,
        suppress_warnings=True,
    )

    ga_instance.run()

    t.close()

    if dim1_data is None:
        dim1_data = np.zeros(xdata.shape)
    if dim2_data is None:
        dim2_data = np.zeros(xdata.shape)
    data = np.array([xdata, dim1_data, dim2_data])

    # return ga_instance

    rendered_formula = render_formula(ga_instance, data, comparison_func)

    return ga_instance, rendered_formula


if __name__ == "__main__":
    from astropy import constants as c

    def comparison_func(x):
        return c.G.value * c.M_sun.value / x ** 2

    algo = run_genetic_algo(comparison_func)
