from typing import List, Callable

import sys
import warnings

# sys.path.append("/home/james/Projects/Astrophysics/formula_finder/")

import numpy as np
import math

__version__ = "0.1.0"


def main(
    comparison_func: Callable = None,
    customParametersData: dict = None,
    total_population_size: int = 1000,
    tree_depth: int = 4,
    num_generations: int = 10000,
    xdata: List[int] = np.linspace(1, 10, 10),
    dim1_data: List[int] = None,
    dim2_data: List[int] = None,
    dimension_names: List[str] = ["x"],
):
    from astropy import constants as c
    from astropy import units as u
    from formula_finder.variables import add_custom_variables

    def default_comparison_func(data):
        x = data[0]
        m1 = data[1]
        m2 = data[2]
        return c.G.value * (m1 * m2) / np.power(x, 2)

    def default_comparison_func(data):
        return math.pi + np.power(data[0], 2) + 5 * data[0]

    def default_comparison_func(data):
        return 1 / np.power(data[0], 2)

    defaultCustomParametersData = {
        # "m1": np.linspace(0.8, 1.2, 10) * c.M_sun.value,
        # "m2": np.linspace(0.8, 1.2, 10) * c.M_earth.value,
    }
    xdata = np.linspace(0.8, 1.3, 10) * 14e9

    if not comparison_func:
        comparison_func = default_comparison_func
    # if not customParametersData:
    #     customParametersData = defaultCustomParametersData

    if len(dimension_names) > 1:
        add_custom_variables(dimension_names[1:])
    from formula_finder.genetic_algo import run_genetic_algo

    return run_genetic_algo(
        comparison_func,
        # customParametersData=customParametersData,
        total_population_size=total_population_size,
        tree_depth=tree_depth,
        num_generations=num_generations,
        xdata=xdata,
        dim1_data=dim1_data,
        dim2_data=dim2_data,
        dimension_names=dimension_names,
    )
    # return algo


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        xdata = np.linspace(0.8, 1.3, 10)

        def ydata(data):
            return np.multiply(np.power(data[0], 3), np.sqrt(data[0]))
            # return np.sqrt(data[0])
            return np.add(4 / np.sqrt(data[0]), np.power(data[0], 3))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, sym = main(
                num_generations=300,
                comparison_func=ydata,
                xdata=xdata,
                # dimension_names=["r"],
            )
        print(sym)
