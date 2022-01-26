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
    # xdata = np.linspace(0.8, 1.3, 10) * 14e9

    if not comparison_func:
        comparison_func = default_comparison_func
    # if not customParametersData:
    #     customParametersData = defaultCustomParametersData

    # if len(dimension_names) > 1:
    #     add_custom_variables(dimension_names[1:])
    # else:
    #     add_custom_variables([])
    from formula_finder.genetic_algo import run_genetic_algo

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


run_formula_finder = main

if __name__ == "__main__":
    from astropy import constants as c
    from astropy import units as u

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # mass_earth = c.M_earth.to(u.kg).value
        # mass_sun = c.M_sun.to(u.kg).value
        # distance_earth_sun = 147e9

        # xdata = np.linspace(0.8, 1.3, 10) * distance_earth_sun
        # dim1_data = np.linspace(0.8, 1.2, 10) * mass_earth
        # dim2_data = np.linspace(0.8, 1.2, 10) * mass_sun

        # xdata = np.linspace(0.8, 1.3, 10)

        # def ydata(data):
        #     return c.G.value * (data[1] * data[2]) / np.power(data[0], 2)
        #     # return np.multiply(np.power(data[0], 3), np.sqrt(data[0]))
        #     # return np.sqrt(data[0])
        #     # return np.add(4 / np.sqrt(data[0]), np.power(data[0], 3))

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     g, sym = main(
        #         num_generations=5,
        #         comparison_func=ydata,
        #         xdata=xdata,
        #         dim1_data=dim1_data,
        #         dim2_data=dim2_data,
        #         dimension_names=["r", "m1", "m2"],
        #     )
        # print(sym)

        # def area_of_circle(data):
        #     return np.pi * np.power(data[0], 2)

        # xdata = np.linspace(0.8, 10, 10)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     g, sym = main(
        #         num_generations=300, comparison_func=area_of_circle, xdata=xdata
        #     )
        # sym
        from matplotlib import pyplot as plt

        def area_of_circle(data):
            return np.pi * np.power(data[0], 2)

        # radius
        earth = {
            "mass": 5.972e24,  # kg
            "orbiting_body_mass": 1.989e30,  # kg
            "distance": 149.6e9,  # m
            "orbital_period": 365.25 * 24 * 3600,  # s
            "orbital_velocity": 29.78e3,  # m/s
        }
        venus = {
            "mass": 4.867e24,  # kg
            "orbiting_body_mass": 1.989e30,  # kg
            "distance": 108.2e9,  # m
            "orbital_period": 224.7 * 24 * 3600,  # s
            "orbital_velocity": 35.02e3,  # m/s
        }
        mars = {
            "mass": 6.39e23,  # kg
            "orbiting_body_mass": 1.989e30,  # kg
            "distance": 227.9e9,  # m
            "orbital_period": 687.0 * 24 * 3600,  # s
            "orbital_velocity": 24.13e3,  # m/s
        }
        jupiter = {
            "mass": 1.898e27,  # kg
            "orbiting_body_mass": 1.989e30,  # kg
            "distance": 778.6e9,  # m
            "orbital_period": 4332.0 * 24 * 3600,  # s
            "orbital_velocity": 13.07e3,  # m/s
        }
        saturn = {
            "mass": 5.683e26,  # kg
            "orbiting_body_mass": 1.989e30,  # kg
            "distance": 1.429e12,  # m
            "orbital_period": 10759.0 * 24 * 3600,  # s
            "orbital_velocity": 9.69e3,  # m/s
        }
        uranus = {
            "mass": 8.683e25,  # kg
            "orbiting_body_mass": 1.989e30,  # kg
            "distance": 2.857e12,  # m
            "orbital_period": 30688.0 * 24 * 3600,  # s
            "orbital_velocity": 6.81e3,  # m/s
        }
        neptune = {
            "mass": 1.024e26,  # kg
            "orbiting_body_mass": 1.989e30,  # kg
            "distance": 4.498e12,  # m
            "orbital_period": 60190.0 * 24 * 3600,  # s
            "orbital_velocity": 5.43e3,  # m/s
        }
        moon = {
            "mass": 7.34767309e22,  # kg
            "orbiting_body_mass": 5.972e24,  # kg
            "distance": 3.844e8,  # m
            "orbital_period": 27.32 * 24 * 3600,  # s
            "orbital_velocity": 1.022e3,  # m/s
        }
        # sun = {
        #     "mass": 1.9891e30,  # kg
        #     "distance_from_sun": 384e6,  # m
        #     "orbital_period": 0,  # s
        #     "orbital_velocity": 1022e3,  # m/s
        # }

        planets = [earth, venus, mars, jupiter, saturn, uranus, neptune, moon]

        xdata = np.array([planet["distance"] for planet in planets])
        dim2_data = np.array([planet["orbiting_body_mass"] for planet in planets])
        dim3_data = np.array([planet["orbital_velocity"] for planet in planets])

        data = [xdata, dim2_data, dim3_data]

        plt.scatter(xdata, dim3_data)

        def orbital_period(data):
            return data[2]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, sym = main(
                num_generations=300,
                comparison_func=orbital_period,
                xdata=xdata,
                dim1_data=dim2_data,
                dim2_data=dim3_data,
                dimension_names=["distance", "mass"],
            )
        sym

        # m/s = m  kg
