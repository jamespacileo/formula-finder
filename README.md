Formula Finder
==========================================

Non-linear curve fitting through use of Genetic Algorithms.

Provide the data and FF will find the best formula for the data.

Examples
---------   

Go to the Jupyter notebook Getting Starter here for a list of examples

Example: Newton's Universal Law of Gravitation (with added noise)
------------------------------------------------------

```
from formula_finder import run_formula_finder

mass_earth = c.M_earth.to(u.kg).value
mass_sun = c.M_sun.to(u.kg).value
distance_earth_sun = 147e9

xdata = np.linspace(0.8, 1.3, 20) * distance_earth_sun
dim1_data = np.linspace(0.8, 1.2, 20) * mass_earth
dim2_data = np.linspace(0.8, 1.2, 20) * mass_sun

def ydata(data):
    # test data with noise
    return c.G.value * (data[1] * data[2]) / np.power(data[0], 2) * np.random.uniform(0.95, 1.05)


g, formula = run_formula_finder(
    num_generations=1000,
    comparison_func=ydata,
    xdata=xdata,
    dim1_data=dim1_data,
    dim2_data=dim2_data,
    dimension_names=["r", "m1", "m2"],
)
formula
>>> 6.6743e-11 * m1 * m2 / r ** 2
```
