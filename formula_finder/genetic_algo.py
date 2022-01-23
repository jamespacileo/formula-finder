import math
from typing import Callable, List
import pygad
import numpy as np
from tqdm.notebook import trange, tqdm
from formula_finder.algo_tree_helpers import convert_array_nodes_to_keys

from formula_finder.binary_tree import (
    number_of_nodes_for_tree,
)
from formula_finder.genetic_helpers import (
    custom_crossover_func,
    custom_mutation,
    fitness_function_factory,
    generate_population,
)

mutation_percent_genes = 10


def run_genetic_algo(
    comparison_func: Callable,
    total_population_size: int = 30,
    tree_depth: int = 3,
    num_generations: int = 1000,
    xdata: List[int] = np.linspace(1, 10, 10),
    customParametersData: dict = None,
):
    num_genes = number_of_nodes_for_tree(tree_depth)
    population = generate_population(total_population_size, num_genes)
    num_parents_mating = math.floor(total_population_size / 20)

    sol_per_pop = 50

    init_range_low = 0
    init_range_high = 20

    parent_selection_type = "sss"
    keep_parents = -1

    crossover_type = "single_point"
    mutation_type = custom_mutation

    fitness_function = fitness_function_factory(
        comparison_func, xdata, customParametersData
    )

    t = tqdm(total=num_generations)

    def callback_gen(ga_instance):
        # print("Generation : ", ga_instance.generations_completed)
        best_solution = ga_instance.best_solution()
        # print("Fitness of the best solution :", best_solution[1])
        # print("Current best solution :", convert_array_nodes_to_keys(best_solution[0]))
        t.update(1)
        t.set_description_str(
            f"Fitness: {best_solution[1]} {convert_array_nodes_to_keys(best_solution[0])}"
        )

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
    )

    ga_instance.run()

    t.close()

    return ga_instance


if __name__ == "__main__":
    from astropy import constants as c

    def comparison_func(x):
        return c.G.value * c.M_sun.value / x ** 2

    algo = run_genetic_algo(comparison_func)
