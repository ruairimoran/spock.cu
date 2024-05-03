import numpy as np


# =====================================================================================================================
# Constraints
# =====================================================================================================================

# --------------------------------------------------------
# Base
# --------------------------------------------------------
class Constraint:
    """
    Base class for constraints
    """
    def __init__(self):
        self.__state_size = None
        self.__control_size = None
        self.__state_matrix = None
        self.__control_matrix = None
        self.__state_matrix_transposed = None
        self.__control_matrix_transposed = None

    def project(self, vector):
        pass

    # GETTERS
    @property
    def is_active(self):
        raise Exception("Base constraint accessed - actual constraint must not be setup")

    @property
    def state_size(self):
        return self.__state_size

    @property
    def control_size(self):
        return self.__control_size

    @property
    def state_matrix(self):
        return self.__state_matrix

    @property
    def control_matrix(self):
        return self.__control_matrix

    @property
    def state_matrix_transposed(self):
        if self.__state_matrix_transposed is None:
            raise Exception("Constraint state matrix transpose called but is None")
        else:
            return self.__state_matrix_transposed

    @property
    def control_matrix_transposed(self):
        if self.__control_matrix_transposed is None:
            raise Exception("Constraint control matrix transpose called but is None")
        else:
            return self.__control_matrix_transposed

    # SETTERS
    @state_size.setter
    def state_size(self, size):
        self.__state_size = size
        # if self.__node_type.is_nonleaf and self.__control_size is not None:
        #     self._set_matrices()
        #     self._get_transpose()
        # elif self.__node_type.is_nonleaf and self.__control_size is None:
        #     pass
        # elif self.__node_type.is_leaf:
        #     self.__control_size = 0
        #     self._set_matrices()
        #     self._get_transpose()
        # else:
        #     raise Exception("Node type missing")

    @control_size.setter
    def control_size(self, size):
        self.__control_size = size
        # if self.__node_type.is_nonleaf and self.__state_size is not None:
        #     self._set_matrices()
        #     self._get_transpose()
        # elif self.__node_type.is_nonleaf and self.__state_size is None:
        #     pass
        # elif self.__node_type.is_leaf:
        #     raise Exception("Attempt to set control size on leaf node")
        # else:
        #     raise Exception("Node type missing")

    def _set_matrices(self):
        pass

    def _get_transpose(self):
        pass
        # if self.__node_type.is_nonleaf:
        #     self.__state_matrix_transposed = np.transpose(self.state_matrix)
        #     self.__control_matrix_transposed = np.transpose(self.control_matrix)
        # elif self.__node_type.is_leaf:
        #     self.__state_matrix_transposed = np.transpose(self.state_matrix)
        # else:
        #     raise Exception("Node type missing")

    @state_matrix.setter
    def state_matrix(self, matrix):
        self.__state_matrix = matrix

    @control_matrix.setter
    def control_matrix(self, matrix):
        pass
        # if self.__node_type.is_nonleaf:
        #     self.__control_matrix = matrix
        # elif self.__node_type.is_leaf:
        #     raise Exception("Attempt to set control constraint matrix of leaf node")
        # else:
        #     raise Exception("Node type missing")


# --------------------------------------------------------
# None
# --------------------------------------------------------
class No(Constraint):
    """
    For no constraints
    """
    def __init__(self):
        super().__init__()

    @property
    def is_active(self):
        return False


# --------------------------------------------------------
# Rectangle
# --------------------------------------------------------
class Rectangle(Constraint):
    """
    A rectangle constraint
    """
    def __init__(self, lowerBound, upperBound):
        """
        :param lowerBound: vector of minimum values
        :param upperBound: vector of maximum values
        """
        super().__init__()
        self._check_constraints(lowerBound, upperBound)
        self.__lb = lowerBound
        self.__ub = upperBound

    @property
    def is_active(self):
        return True

    def _set_matrices(self):
        self.state_matrix = np.vstack((np.eye(self.state_size), np.zeros((self.control_size, self.state_size))))
        # if self._Constraint__node_type.is_nonleaf:
        #     self.control_matrix = np.vstack((np.zeros((self.state_size, self.control_size)), np.eye(self.control_size)))

    def project(self, vector):
        self._check_input(vector)
        constrained_vector = np.zeros(vector.shape)
        for i in range(vector.size):
            constrained_vector[i] = self._constrain(vector[i], self.__lb[i], self.__ub[i])

        return constrained_vector

    @staticmethod
    def _check_constraints(lb, ub):
        if lb.size != ub.size:
            raise Exception("Rectangle constraint - min and max vectors sizes are not equal")
        for i in range(lb.size):
            if lb[i] is None and ub[i] is None:
                raise Exception("Rectangle constraint - both min and max constraints cannot be None")
            if lb[i] is None or ub[i] is None:
                pass
            else:
                if lb[i] > ub[i]:
                    raise Exception("Rectangle constraint - min greater than max")

    @staticmethod
    def _constrain(value, mini, maxi):
        if mini <= value <= maxi:
            return value
        elif value <= mini:
            return mini
        elif value >= maxi:
            return maxi
        else:
            raise ValueError(f"Rectangle constraint - '{value}' value cannot be constrained")

    def _check_input(self, vector):
        if vector.size != self.state_matrix.shape[0]:
            raise Exception("Rectangle constraint - input vector does not equal expected size")


# =====================================================================================================================
# Risks
# =====================================================================================================================

# --------------------------------------------------------
# Base
# --------------------------------------------------------
class CoherentRisk:
    """
    Base class for coherent risks.
    """
    def __init__(self):
        """
        Ambiguity sets of coherent risk measures can be expressed by conic inequalities,
        defined by a tuple (E, F, cone, b).
        """
        self.__num_children = None
        self.__children_probabilities = None
        self.__matrix_e = None  # coefficient matrix of mu
        self.__matrix_f = None  # coefficient matrix of nu
        self.__cone = None
        self.__vector_b = None

    def project(self, vector):
        pass

    # GETTERS
    @property
    def is_active(self):
        raise Exception("Base constraint accessed - actual constraint must not be setup")

    @property
    def state_size(self):
        return self.__state_size

    @property
    def control_size(self):
        return self.__control_size

    @property
    def state_matrix(self):
        return self.__state_matrix

    @property
    def control_matrix(self):
        return self.__control_matrix

    @property
    def state_matrix_transposed(self):
        if self.__state_matrix_transposed is None:
            raise Exception("Constraint state matrix transpose called but is None")
        else:
            return self.__state_matrix_transposed

    @property
    def control_matrix_transposed(self):
        if self.__control_matrix_transposed is None:
            raise Exception("Constraint control matrix transpose called but is None")
        else:
            return self.__control_matrix_transposed

    # SETTERS
    @state_size.setter
    def state_size(self, size):
        self.__state_size = size
        # if self.__node_type.is_nonleaf and self.__control_size is not None:
        #     self._set_matrices()
        #     self._get_transpose()
        # elif self.__node_type.is_nonleaf and self.__control_size is None:
        #     pass
        # elif self.__node_type.is_leaf:
        #     self.__control_size = 0
        #     self._set_matrices()
        #     self._get_transpose()
        # else:
        #     raise Exception("Node type missing")

    @control_size.setter
    def control_size(self, size):
        self.__control_size = size
        # if self.__node_type.is_nonleaf and self.__state_size is not None:
        #     self._set_matrices()
        #     self._get_transpose()
        # elif self.__node_type.is_nonleaf and self.__state_size is None:
        #     pass
        # elif self.__node_type.is_leaf:
        #     raise Exception("Attempt to set control size on leaf node")
        # else:
        #     raise Exception("Node type missing")

    def _set_matrices(self):
        pass

    def _get_transpose(self):
        pass
        # if self.__node_type.is_nonleaf:
        #     self.__state_matrix_transposed = np.transpose(self.state_matrix)
        #     self.__control_matrix_transposed = np.transpose(self.control_matrix)
        # elif self.__node_type.is_leaf:
        #     self.__state_matrix_transposed = np.transpose(self.state_matrix)
        # else:
        #     raise Exception("Node type missing")

    @state_matrix.setter
    def state_matrix(self, matrix):
        self.__state_matrix = matrix

    @control_matrix.setter
    def control_matrix(self, matrix):
        pass
        # if self.__node_type.is_nonleaf:
        #     self.__control_matrix = matrix
        # elif self.__node_type.is_leaf:
        #     raise Exception("Attempt to set control constraint matrix of leaf node")
        # else:
        #     raise Exception("Node type missing")


# --------------------------------------------------------
# Average Value at Risk
# --------------------------------------------------------
class AVaR(CoherentRisk):
    """
    Risk item: Average Value at Risk class
    """

    def __init__(self, alpha):
        """
        :param alpha: AVaR risk parameter

        Note: ambiguity sets of coherent risk measures can be expressed by conic inequalities,
                defined by a tuple (E, F, cone, b)
        """
        super().__init__()
        self.__num_children = None
        self.__children_probabilities = None
        if 0 <= alpha <= 1:
            self.__alpha = alpha
        else:
            raise ValueError("alpha value '%d' not supported" % alpha)
        self.__matrix_e = None  # coefficient matrix of mu
        self.__matrix_f = None  # coefficient matrix of nu
        self.__cone = None
        self.__vector_b = None

    def _make_e_f_cone_b(self):
        eye = np.eye(self.__num_children)
        self.__matrix_e = np.vstack((self.__alpha*eye, -eye, np.ones((1, self.__num_children))))
        self.__matrix_f = np.zeros((2 * self.__num_children + 1, 0))
        # self.__cone = core_cones.Cartesian([core_cones.NonnegativeOrthant(dimension=2 * self.__num_children),
        #                                     core_cones.Zero(dimension=1)])
        self.__vector_b = np.vstack((np.asarray(self.__children_probabilities).reshape(-1, 1),
                                     np.zeros((self.__num_children, 1)), 1))

    # GETTERS
    @property
    def is_risk(self):
        return True

    @property
    def alpha(self):
        """AVaR risk parameter alpha"""
        return self.__alpha

    @property
    def matrix_e(self):
        """Ambiguity set matrix E"""
        return self.__matrix_e

    @property
    def matrix_f(self):
        """Ambiguity set matrix F"""
        return self.__matrix_f

    @property
    def cone(self):
        """Ambiguity set cone"""
        return self.__cone

    @property
    def vector_b(self):
        """Ambiguity set vector b"""
        return self.__vector_b

    @property
    def probs(self):
        return self.__children_probabilities

    # SETTERS
    @probs.setter
    def probs(self, vector):
        self.__children_probabilities = vector
        self.__num_children = vector.size
        self._make_e_f_cone_b()
