from formula_finder import __version__
from formula_finder.algo_tree_helpers import convert_array_nodes_to_keys
from formula_finder.binary_tree import print_binary_tree
from formula_finder.genetic_algo import run_genetic_algo
from formula_finder.variables import add_custom_variables


def test_version():
    assert __version__ == "0.1.0"


def test_ga_run():
    def comparison_func(x):
        return c.G.value * c.M_sun.value / x ** 2

    algo = run_genetic_algo(comparison_func)


add_custom_variables(["m1", "m2"])

solution = convert_array_nodes_to_keys(
    [4, 6, 4, 2, 24, 24, 0, 22, 3, 20, 21, 20, 23, 0, 23]
)
print_binary_tree(solution)
