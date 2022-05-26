import numpy as np

from .mcda_method_smaa import MCDA_method_smaa
from ..additions import rank_preferences


class VIKOR_SMAA(MCDA_method_smaa):
    def __init__(self, normalization_method = None, v = 0.5):
        """Create the VIKOR method object.

        Parameters
        -----------

            normalization_method : function
                VIKOR does not use normalization by default, thus `normalization_method` is set to None by default.
                However, you can choose method for normalization of decision matrix chosen `normalization_method` from `normalizations`.
                It is used in a way `normalization_method(X, types)` where `X` is a decision matrix
                and `types` is a vector with criteria types where 1 means profit and -1 means cost.
            v : float
                parameter that is the weight of strategy of the majority of criteria (the maximum group utility)
        """
        self.v = v
        self.normalization_method = normalization_method

    def __call__(self, matrix, types, iterations):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.
        
        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            types : ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.
            iterations : int
                Number of iterations of SMAA
        
        Returns
        --------
            ndrarray, ndarray
                Matrix with acceptability indexes values for each alternative in rows in relation to each rank in columns,
                Matrix with central weight vectors for each alternative in rows
        
        Examples
        ---------
        >>> vikor_smaa = VIKOR_SMAA(normalization_method = minmax_normalization)
        >>> acceptability_index, central_weights = vikor_smaa(matrix, types, iterations = 10000)
        """

        VIKOR_SMAA._verify_input_data(matrix, types, iterations)
        return VIKOR_SMAA._vikor_smaa(self, matrix, types, iterations, self.normalization_method, self.v)

    def _generate_weights(self, n):
        # n weight generation - when no preference information available
        # generate n - 1 uniform distributed weights within the range [0, 1]
        w = np.random.uniform(0, 1, n)

        # sort weights into ascending order (q[1], ..., q[n-1])
        ind = np.argsort(w)
        w = w[ind]

        # insert 0 as the first q[0] and 1 as the last (q[n]) numbers
        w = np.insert(w, 0, 0)
        w = np.insert(w, len(w), 1)

        # the weights are obtained as intervals between consecutive numbers (w[j] = q[j] - q[j-1])
        weights = [w[i] - w[i - 1] for i in range(1, n + 1)]
        weights = np.array(weights)

        # scale the generated weights so that their sum is 1
        new_weights = weights / np.sum(weights)
        return new_weights


    @staticmethod
    def _vikor_smaa(self, matrix, types, iterations, normalization_method, v):
        m, n = matrix.shape

        # Central weight vector for each alternative
        central_weights = np.zeros((m, n))
        
        # Acceptability index of each place for each alternative
        acceptability_index = np.zeros((m, m))

        for it in range(iterations):
            # generate weights
            weights = self._generate_weights(n)
        
            # run the VIKOR algorithm
            # Without applying a special normalization method
            if normalization_method == None:

                # Determine the best `fstar` and the worst `fmin` values of all criterion function
                maximums_matrix = np.amax(matrix, axis = 0)
                minimums_matrix = np.amin(matrix, axis = 0)

                fstar = np.zeros(matrix.shape[1])
                fmin = np.zeros(matrix.shape[1])

                # for profit criteria (`types` == 1) and for cost criteria (`types` == -1)
                fstar[types == 1] = maximums_matrix[types == 1]
                fstar[types == -1] = minimums_matrix[types == -1]
                fmin[types == 1] = minimums_matrix[types == 1]
                fmin[types == -1] = maximums_matrix[types == -1]

                weighted_matrix = weights * ((fstar - matrix) / (fstar - fmin))
            else:
                # With applying the special normalization method
                norm_matrix = normalization_method(matrix, types)
                fstar = np.amax(norm_matrix, axis = 0)
                fmin = np.amin(norm_matrix, axis = 0)
                weighted_matrix = weights * ((fstar - norm_matrix) / (fstar - fmin))

            # Calculate the `S` and `R` values
            S = np.sum(weighted_matrix, axis = 1)
            R = np.amax(weighted_matrix, axis = 1)
            # Calculate the Q values
            Sstar = np.min(S)
            Smin = np.max(S)
            Rstar = np.min(R)
            Rmin = np.max(R)
            # Calculate VIKOR preference values
            Q = v * (S - Sstar) / (Smin - Sstar) + (1 - v) * (R - Rstar) / (Rmin - Rstar)
            # Rank alternatives according to VIKOR preference values in ascending order
            rank = rank_preferences(Q, reverse = False)

            # add value for the acceptability index for each alternative considering rank
            for el, r in enumerate(rank):
                acceptability_index[el, r - 1] += 1

            # add central weights for the best scored alternative
            ind_min = np.argmin(Q)
            central_weights[ind_min, :] += weights

        #
        # Calculate the acceptability index
        acceptability_index = acceptability_index / iterations

        # Calculate central the weights vectors
        central_weights = central_weights / iterations
        for i in range(m):
            if np.sum(central_weights[i, :]):
                central_weights[i, :] = central_weights[i, :] / np.sum(central_weights[i, :])

        return acceptability_index, central_weights