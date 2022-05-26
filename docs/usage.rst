Usage
======

.. _installation:

Installation
-------------

To use ``objective_weighting`` package, first install it using pip:

.. code-block:: python

	pip install objective-weighting


Usage examples
----------------------


The VIKOR method
__________________

.. code-block:: python

	import numpy as np
	from objective_weighting.mcda_methods import VIKOR
	from objective_weighting.additions import rank_preferences

	# Provide decision matrix in array numpy.darray.
	matrix = np.array([[8, 7, 2, 1],
	[5, 3, 7, 5],
	[7, 5, 6, 4],
	[9, 9, 7, 3],
	[11, 10, 3, 7],
	[6, 9, 5, 4]])

	# Provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.4, 0.3, 0.1, 0.2])

	# Provide criteria types in array numpy.darray. Profit criteria are represented by 1, and cost criteria by -1.
	types = np.array([1, 1, 1, 1])

	# Create the VIKOR method object providing `v` parameter. The default `v` parameter is set to 0.5, so if you do not provide it, `v` will be equal to 0.5.
	vikor = VIKOR(v = 0.625)

	# Calculate the VIKOR preference values of alternatives.
	pref = vikor(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives ascendingly according to the VIKOR algorithm (reverse = False means sorting in ascending order) according to preference values.
	rank = rank_preferences(pref, reverse = False)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.6399 1.     0.6929 0.2714 0.     0.6939]
	Ranking:  [3 6 4 2 1 5]
	

Correlation coefficents
__________________________

Spearman correlation coefficient

.. code-block:: python

	import numpy as np
	from objective_weighting import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods.
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `spearman` coefficient.
	coeff = corrs.spearman(R, Q)
	print('Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Spearman coeff:  0.9

	
	
Weighted Spearman correlation coefficient

.. code-block:: python

	import numpy as np
	from objective_weighting import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods.
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `weighted_spearman` coefficient.
	coeff = corrs.weighted_spearman(R, Q)
	print('Weighted Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Weighted Spearman coeff:  0.8833

	
	
Pearson correlation coefficient

.. code-block:: python

	import numpy as np
	from objective_weighting import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods.
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `pearson_coeff` coefficient.
	coeff = corrs.pearson_coeff(R, Q)
	print('Pearson coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Pearson coeff:  0.9
	
	
	
Methods for criteria weights determination
___________________________________________

Entropy weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights

	matrix = np.array([[30, 30, 38, 29],
	[19, 54, 86, 29],
	[19, 15, 85, 28.9],
	[68, 70, 60, 29]])

	weights = mcda_weights.entropy_weighting(matrix)

	print('Entropy weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Entropy weights:  [0.463  0.3992 0.1378 0.    ]
	

CRITIC weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights

	matrix = np.array([[5000, 3, 3, 4, 3, 2],
	[680, 5, 3, 2, 2, 1],
	[2000, 3, 2, 3, 4, 3],
	[600, 4, 3, 1, 2, 2],
	[800, 2, 4, 3, 3, 4]])

	weights = mcda_weights.critic_weighting(matrix)

	print('CRITIC weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	CRITIC weights:  [0.157  0.2495 0.1677 0.1211 0.1541 0.1506]


Standard deviation weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights

	matrix = np.array([[0.619, 0.449, 0.447],
	[0.862, 0.466, 0.006],
	[0.458, 0.698, 0.771],
	[0.777, 0.631, 0.491],
	[0.567, 0.992, 0.968]])

	weights = mcda_weights.std_weighting(matrix)

	print('Standard deviation weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Standard deviation weights:  [0.2173 0.2945 0.4882]
	
	
Equal weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights

	matrix = np.array([[0.619, 0.449, 0.447],
	[0.862, 0.466, 0.006],
	[0.458, 0.698, 0.771],
	[0.777, 0.631, 0.491],
	[0.567, 0.992, 0.968]])
	
	weights = mcda_weights.equal_weighting(matrix)
	print('Equal weights: ', np.round(weights, 3))
	
Output

.. code-block:: console
	
	Equal weights:  [0.333 0.333 0.333]


Gini coefficient-based weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights
	
	matrix = np.array([[29.4, 83, 47, 114, 12, 30, 120, 240, 170, 90, 1717.75],
	[30, 38.1, 124.7, 117, 16, 60, 60, 60, 93, 70, 2389],
	[29.28, 59.27, 41.13, 58, 16, 30, 60, 120, 170, 78, 239.99],
	[33.6, 71, 55, 159, 23.6, 60, 240, 240, 132, 140, 2099],
	[21, 59, 41, 66, 16, 24, 60, 120, 170, 70, 439],
	[35, 65, 42, 134, 12, 60, 240, 240, 145, 60, 1087],
	[47, 79, 54, 158, 19, 60, 120, 120, 360, 72, 2499],
	[28.3, 62.3, 44.9, 116, 12, 30, 60, 60, 130, 90, 999.99],
	[36.9, 28.6, 121.6, 130, 12, 60, 120, 120, 80, 80, 1099],
	[32, 59, 41, 60, 16, 30, 120, 120, 170, 60, 302.96],
	[28.4, 66.3, 48.6, 126, 12, 60, 240, 240, 132, 135, 1629],
	[29.8, 46, 113, 47, 18, 50, 50, 50, 360, 72, 2099],
	[20.2, 64, 80, 70, 8, 24, 60, 120, 166, 480, 699.99],
	[33, 60, 44, 59, 12, 30, 60, 120, 170, 90, 388],
	[29, 59, 41, 55, 16, 30, 60, 120, 170, 120, 299],
	[29, 59, 41, 182, 12, 30, 30, 60, 94, 140, 249],
	[29.8, 59.2, 41, 65, 16, 30, 60, 120, 160, 90, 219.99],
	[28.8, 62.5, 41, 70, 12, 60, 120, 120, 170, 138, 1399.99],
	[24, 40, 59, 60, 12, 10, 30, 30, 140, 78, 269.99],
	[30, 60, 45, 201, 16, 30, 30, 30, 170, 90, 199.99]])

	weights = mcda_weights.gini_weighting(matrix)
	print('Gini coefficient-based weights: ', np.round(weights, 4))


Output

.. code-block:: console

	Gini coefficient-based weights:  [0.0362 0.0437 0.0848 0.0984 0.048  0.0842 0.1379 0.1125 0.0745 0.1107 0.169 ]


MEREC weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights
	
	matrix = np.array([[450, 8000, 54, 145],
	[10, 9100, 2, 160],
	[100, 8200, 31, 153],
	[220, 9300, 1, 162],
	[5, 8400, 23, 158]])
	
	types = np.array([1, 1, -1, -1])

	weights = mcda_weights.merec_weighting(matrix, types)
	print('MEREC weights: ', np.round(weights, 4))


Output

.. code-block:: console

	MEREC weights:  [0.5752 0.0141 0.4016 0.0091]


Statistical variance weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights
	
	matrix = np.array([[0.619, 0.449, 0.447],
	[0.862, 0.466, 0.006],
	[0.458, 0.698, 0.771],
	[0.777, 0.631, 0.491],
	[0.567, 0.992, 0.968]])
	
	weights = mcda_weights.stat_var_weighting(matrix)
	print('Statistical variance weights: ', np.round(weights, 4))


Output

.. code-block:: console

	Statistical variance weights:  [0.3441 0.3497 0.3062]


CILOS weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights

	matrix = np.array([[3, 100, 10, 7],
	[2.500, 80, 8, 5],
	[1.800, 50, 20, 11],
	[2.200, 70, 12, 9]])

	types = np.array([-1, 1, -1, 1])

	weights = mcda_weights.cilos_weighting(matrix, types)
	print('CILOS weights: ', np.round(weights, 3))


Output

.. code-block:: console

	CILOS weights:  [0.334 0.22  0.196 0.25 ]


IDOCRIW weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights
	
	matrix = np.array([[3.0, 100, 10, 7],
	[2.5, 80, 8, 5],
	[1.8, 50, 20, 11],
	[2.2, 70, 12, 9]])

	types = np.array([-1, 1, -1, 1])

	weights = mcda_weights.idocriw_weighting(matrix, types)
	print('IDOCRIW weights: ', np.round(weights, 3))

Output

.. code-block:: console

	IDOCRIW weights:  [0.166 0.189 0.355 0.291]
	

Angle weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights
	
	matrix = np.array([[30, 30, 38, 29],
	[19, 54, 86, 29],
	[19, 15, 85, 28.9],
	[68, 70, 60, 29]])

	types = np.array([1, 1, 1, 1])

	weights = mcda_weights.angle_weighting(matrix, types)
	print('Angle weights: ', np.round(weights, 4))


Output

.. code-block:: console

	Angle weights:  [0.415  0.3612 0.2227 0.0012]


Coefficient of variation weighting method
		
.. code-block:: python

	import numpy as np
	from objective_weighting import weighting_methods as mcda_weights
	
	matrix = np.array([[30, 30, 38, 29],
	[19, 54, 86, 29],
	[19, 15, 85, 28.9],
	[68, 70, 60, 29]])

	weights = mcda_weights.coeff_var_weighting(matrix)
	print('Coefficient of variation weights: ', np.round(weights, 4))


Output

.. code-block:: console

	Coefficient of variation weights:  [0.4258 0.361  0.2121 0.0011]
	
	
Stochastic Multicriteria Acceptability Analysis Method - SMAA (VIKOR_SMAA)
_______________________________________________________________________________

.. code-block:: python

	from objective_weighting.mcda_methods import VIKOR_SMAA

	vikor_smaa = VIKOR_SMAA()
	acceptability_index, central_weights = vikor_smaa(matrix, types, iterations = 10000)
	
	
	
Normalization methods
______________________

Here is an example of ``vector_normalization`` usage. Other normalizations provided in module ``normalizations``, namely ``minmax_normalization``, ``max_normalization``,
``sum_normalization``, ``linear_normalization`` are used in analogous way.


Vector normalization

.. code-block:: python

	import numpy as np
	from objective_weighting import normalizations as norms

	matrix = np.array([[8, 7, 2, 1],
	[5, 3, 7, 5],
	[7, 5, 6, 4],
	[9, 9, 7, 3],
	[11, 10, 3, 7],
	[6, 9, 5, 4]])

	types = np.array([1, 1, 1, 1])

	norm_matrix = norms.vector_normalization(matrix, types)
	print('Normalized matrix: ', np.round(norm_matrix, 4))
	
Output

.. code-block:: console

	Normalized matrix:  [[0.4126 0.3769 0.1525 0.0928]
	 [0.2579 0.1615 0.5337 0.4642]
	 [0.361  0.2692 0.4575 0.3714]
	 [0.4641 0.4845 0.5337 0.2785]
	 [0.5673 0.5384 0.2287 0.6499]
	 [0.3094 0.4845 0.3812 0.3714]]
