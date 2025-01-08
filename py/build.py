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
        self.__dim = 0

    # TYPES
    @property
    def is_no(self):
        return False

    @property
    def is_rectangle(self):
        return False

    @property
    def is_polyhedron(self):
        return False

    @property
    def is_ball(self):
        return False

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, d):
        self.__dim = d

    def assign_dual(self, dual, idx, num_vec):
        pass

    def op_nonleaf(self, x, n, u):
        pass

    def op_leaf(self, x, n):
        pass

    def adj_nonleaf(self, dual, x, n, u):
        pass

    def adj_leaf(self, dual, x, n):
        pass


# --------------------------------------------------------
# None
# --------------------------------------------------------
class No(Constraint):
    """
    For no constraints
    """

    def __init__(self):
        super().__init__()
        self.__nada = np.array([]).reshape(1, 0)

    @property
    def is_no(self):
        return True

    def assign_dual(self, dual, idx, num_vec):
        return self.__nada, idx

    def op_nonleaf(self, x, n, u):
        return self.__nada

    def op_leaf(self, x, n):
        return self.__nada

    def adj_nonleaf(self, dual, x, n, u):
        return x, u

    def adj_leaf(self, dual, x, n):
        return x


# --------------------------------------------------------
# Rectangle
# --------------------------------------------------------
class Rectangle(Constraint):
    """
    A rectangle constraint
    """

    def __init__(self, lower_bound, upper_bound):
        """
        :param lower_bound: vector of minimum values
        :param upper_bound: vector of maximum values
        """
        super().__init__()
        self.__check_bounds(lower_bound, upper_bound)
        self.__lb = np.array(lower_bound).reshape(-1, 1)
        self.__ub = np.array(upper_bound).reshape(-1, 1)
        self.dim = self.__ub.size

    @property
    def is_rectangle(self):
        return True

    @property
    def lower_bound(self):
        return self.__lb

    @property
    def upper_bound(self):
        return self.__ub

    @staticmethod
    def __check_bounds(lb, ub):
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

    def assign_dual(self, dual, idx, num_vec):
        d = [np.array(dual[idx + i * self.dim:idx + i * self.dim + self.dim]).reshape(self.dim, 1) for i in
             range(num_vec)]
        idx += self.dim * num_vec
        return d, idx

    def op_nonleaf(self, x, n, u):
        dual = [np.zeros((x[0].size + u[0].size, 1))] * n
        for i in range(n):
            dual[i] = np.vstack((x[i], u[i]))
        return dual

    def op_leaf(self, x, n):
        dual = [np.zeros((x[0].size, 1))] * n
        for i in range(n):
            dual[i] = x[i]
        return dual

    def adj_nonleaf(self, dual, x, n, u):
        dim = x[0].size
        for i in range(n):
            x[i] = dual[i][:dim]
            u[i] = dual[i][dim:]
        return x, u

    def adj_leaf(self, dual, x, n):
        for i in range(n):
            x[i] = dual[i]
        return x


# --------------------------------------------------------
# Polyhedron
# --------------------------------------------------------
class Polyhedron(Constraint):
    """
    A polyhedral constraint of the form: Ax <= b
    """

    def __init__(self, matrix, upper_bound):
        """
        :param matrix: pre-multiplying matrix
        :param upper_bound: vector
        """
        super().__init__()
        self.__check_arguments(matrix, upper_bound)
        self.__matrix = matrix
        self.__ub = np.array(upper_bound).reshape(-1, 1)
        self.dim = self.__ub.size

    @property
    def is_polyhedron(self):
        return True

    @property
    def matrix(self):
        return self.__matrix

    @property
    def upper_bound(self):
        return self.__ub

    @staticmethod
    def __check_arguments(A, b):
        if not isinstance(A, np.ndarray):
            raise Exception("Polyhedron constraint - matrix is not a numpy array")
        if A.shape[0] != b.size:
            raise Exception("Polyhedron constraint - matrix row and upper bound dimensions are not equal")
        for i in range(b.size):
            if b[i] is None:
                raise Exception("Polyhedron constraint - bound cannot be None")

    def assign_dual(self, dual, idx, num_vec):
        d = [np.array(dual[idx + i * self.dim:idx + i * self.dim + self.dim]).reshape(self.dim, 1) for i in
             range(num_vec)]
        idx += self.dim * num_vec
        return d, idx

    def op_nonleaf(self, x, n, u):
        dual = [np.zeros((self.__matrix.shape[0], 1))] * n
        for i in range(n):
            dual[i] = self.matrix @ np.vstack((x[i], u[i]))
        return dual

    def op_leaf(self, x, n):
        dual = [np.zeros((self.__matrix.shape[0], 1))] * n
        for i in range(n):
            dual[i] = self.matrix @ x[i]
        return dual

    def adj_nonleaf(self, dual, x, n, u):
        dim = x[0].size
        for i in range(n):
            xu = self.matrix.T @ dual[i]
            x[i] = xu[:dim]
            u[i] = xu[dim:]
        return x, u

    def adj_leaf(self, dual, x, n):
        for i in range(n):
            x[i] = self.matrix.T @ dual[i]
        return x


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
        self._children_probabilities = None
        self._num_children = None
        self._matrix_e = None
        self._matrix_f = None
        self._cone_k_dimension = None
        self._vector_b = None

    def make_risk(self, conditional_probabilities_of_children):
        pass

    # GETTERS
    @property
    def is_avar(self):
        return False

    @property
    def is_evar(self):
        return False

    @property
    def e(self):
        return self._matrix_e

    @property
    def f(self):
        return self._matrix_f

    @property
    def k(self):
        return self._cone_k_dimension

    @property
    def b(self):
        return self._vector_b


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
        if 0 <= alpha <= 1:
            self.__alpha = alpha
        else:
            raise ValueError("AVaR alpha must be between 0 and 1; value '%d' not valid" % alpha)

    def make_risk(self, conditional_probabilities_of_children):
        self._children_probabilities = conditional_probabilities_of_children
        self._num_children = conditional_probabilities_of_children.size
        self.__make_e_f_k_b()
        return self

    def __make_e_f_k_b(self):
        eye = np.eye(self._num_children)
        self._matrix_e = np.vstack((self.__alpha * eye,
                                    -eye,
                                    np.ones((1, self._num_children))))
        # Matrix F not applicable for AVaR
        self._cone_k_dimension = self._num_children * 2 + 1
        self._vector_b = np.vstack((np.asarray(self._children_probabilities).reshape(-1, 1),
                                    np.zeros((self._num_children, 1)), 1))
        return self

    # GETTERS
    @property
    def is_avar(self):
        return True

    @property
    def alpha(self):
        return self.__alpha
