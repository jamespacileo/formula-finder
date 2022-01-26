Formula Finder - Alpha
==========================================

Non-linear curve fitting through use of Genetic Algorithms.

*Feed it the data* -> *Pull out the formula*

Why build this?
------------

I've recently been working on a small research paper, and wondered if GA could smartly speed up the process of Formula Hacking.

To my surprise, this initial prototype is able to extrapolate a few known formulas fairly quickly. So I thought I might just open source this.

Hope you find this useful :)

Examples
---------   

Go to the Jupyter notebook [Getting Started](https://github.com/jamespacileo/formula-finder/blob/master/notebooks/Getting%20Started.ipynb) here for a list of examples

Example: Newton's Universal Law of Gravitation (with added noise)
------------------------------------------------------

Here below we mock some test data to pass to the algorithm, with some added noise.

```python
from formula_finder import run_formula_finder

mass_earth = c.M_earth.to(u.kg).value # in kg
mass_sun = c.M_sun.to(u.kg).value # 
distance_earth_sun = 147e9 # in meters

xdata = np.linspace(0.8, 1.3, 20) * distance_earth_sun # x axis for distance
dim1_data = np.linspace(0.8, 1.2, 20) * mass_earth # variations of mass
dim2_data = np.linspace(0.8, 1.2, 20) * mass_sun # variations of mass

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

TODOS:
----------

- Better strategies at preventing evolution staleness
- Explore RNNs to either replace or add to the algorithm
- Usage on practical cases using either astrophyics or other areas where curve fitting is needed

Please leave feedback :)