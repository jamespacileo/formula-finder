from formula_finder import __version__
from formula_finder.genetic_algo import run_genetic_algo


def test_version():
    assert __version__ == "0.1.0"


def test_ga_run():
    def comparison_func(x):
        return c.G.value * c.M_sun.value / x ** 2

    algo = run_genetic_algo(comparison_func)
