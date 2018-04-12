import numpy as np
from typing import List                   # for writing function defs

def values_for_simulation(values: List[float], sims_per_val: int) -> List[float]:
    """Return an array that repeats values by sims_per_val

    >>> values_for_simulation([0, 0.5, 5], 3)
    array([[ 0. ],
           [ 0. ],
           [ 0. ],
           [ 0.5],
           [ 0.5],
           [ 0.5],
           [ 5. ],
           [ 5. ],
           [ 5. ]])
    """

    values_array = np.repeat(values, sims_per_val) # duplicate values * simsPerVal
    values_array = values_array.reshape(len(values_array),1) # make 2D array

    # PRINT SUMMARY
    print("You will run {0} simulations for each of {1} values for a total of {2} simulations.".format(sims_per_val, len(values), values_array.shape[0]))
    print("This will take about {0}.".format("[figure out how to calculate]"))

    return values_array

def parameter_values(scaling_values, upper_boundary):
