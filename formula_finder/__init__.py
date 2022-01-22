from typing import List, Callable

import sys
import warnings

# sys.path.append("/home/james/Projects/Astrophysics/formula_finder/")

import numpy as np


__version__ = "0.1.0"


def main(
    comparison_func: Callable = None,
    customParametersData: dict = None,
    total_population_size: int = 30,
    tree_depth: int = 3,
    num_generations: int = 1000,
    xdata: List[int] = np.linspace(1, 10, 10),
):
    from astropy import constants as c
    from astropy import units as u
    from formula_finder.variables import add_custom_variables

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

    if not comparison_func:
        comparison_func = default_comparison_func
    if not customParametersData:
        customParametersData = defaultCustomParametersData

    add_custom_variables(customParametersData.keys())
    from formula_finder.genetic_algo import run_genetic_algo

    algo = run_genetic_algo(
        comparison_func,
        customParametersData=customParametersData,
        total_population_size=total_population_size,
        tree_depth=tree_depth,
        num_generations=num_generations,
        xdata=xdata,
    )
    return algo


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
