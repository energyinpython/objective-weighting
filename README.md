# objective-weighting
The Python 3 Library of Objective Weighting Techniques for MCDA methods.


## Installation

```
pip install objective-weighting
```

## Usage

`objective-weighting` is the Python 3 package that provides 11 objective weighting methods, which can be used to determine criteria weights for 
solving multi-criteria problems with Multi-Criteria Decision Analysis (MCDA) methods. The first step is providing the decision matrix `matrix` with alternatives 
performance values. The decision matrix is two-dimensional and contains m alternatives in rows and n criteria in columns. You also have to provide 
criteria types `types`. Criteria types are equal to 1 for profit criteria and -1 for cost criteria. Then you have to calculate criteria weights 
using chosen from `weighting_methods` module weighting method. Depending on the method chosen, you have to provide `matrix` or `matrix` and `types` as 
weighting method arguments. It is detailed in Usage in the documentation. Then you can evaluate alternatives from the decision matrix using the VIKOR method 
from `mcda_methods` module. The VIKOR method returns a vector with preference values `pref` assigned to alternatives. To rank alternatives 
according to VIKOR preference values, you have to sort them in ascending order because, in the VIKOR method, the best alternative has the lowest 
preference value. The alternatives are ranked using the `rank_preferences` method provided in the `additions` module of the `objective-weighting` 
package. Parameter `reverse = False` means that alternatives 
are sorted in ascending order. Here is an example of using the Entropy weighting method `entropy_weighting` for determining criteria weights and 
the VIKOR method to calculate preference values:

```python
import numpy as np
from objective_weighting.mcda_methods import VIKOR
from objective_weighting import weighting_methods as mcda_weights
from objective_weighting import normalizations as norms
from objective_weighting.additions import rank_preferences

matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
[256, 8, 32, 1.0, 1.8, 6919.99],
[256, 8, 53, 1.6, 1.9, 8400],
[256, 8, 41, 1.0, 1.75, 6808.9],
[512, 8, 35, 1.6, 1.7, 8479.99],
[256, 4, 35, 1.6, 1.7, 7499.99]])

types = np.array([1, 1, 1, 1, -1, -1])
weights = mcda_weights.entropy_weighting(matrix)

# Create the VIKOR method object
vikor = VIKOR(normalization_method=norms.minmax_normalization)
# Calculate alternatives preference function values with VIKOR method
pref = vikor(matrix, weights, types)
# Rank alternatives according to preference values
rank = rank_preferences(pref, reverse = False)
```

## License

`objective-weighting` was created by Aleksandra Bączkiewicz. It is licensed under the terms of the MIT license.

## Documentation

Documentation of this library with instruction for installation and usage is 
provided [here](https://objective-weighting.readthedocs.io/en/latest/)
