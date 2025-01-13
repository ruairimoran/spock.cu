import numpy as np


# =====================================================================================================================
# Dynamics
# =====================================================================================================================

# --------------------------------------------------------
# Base
# --------------------------------------------------------
class Dynamics:
    """
    Base class for dynamics
    """

    def __init__(self, state_dyn, input_dyn, affine_dyn):
        self.__state_dyn_matrix = state_dyn
        self.__input_dyn_matrix = input_dyn
        self.__affine_dyn_vector = affine_dyn
        self.__state_input_dyn_matrix = np.hstack((self.__state_dyn_matrix, self.__input_dyn_matrix))

    # TYPES
    @property
    def is_linear(self):
        return False

    @property
    def is_affine(self):
        return False

    @property
    def state(self):
        return self.__state_dyn_matrix

    @property
    def input(self):
        return self.__input_dyn_matrix

    @property
    def affine(self):
        return self.__affine_dyn_vector

    @property
    def state_input(self):
        return self.__state_input_dyn_matrix


# --------------------------------------------------------
# Linear
# --------------------------------------------------------
class Linear(Dynamics):
    """
    Linear dynamics
    """

    def __init__(self, state_dyn, input_dyn):
        super().__init__(state_dyn, input_dyn, np.zeros((state_dyn.shape[0], 1)))

    # TYPES
    @property
    def is_linear(self):
        return True


# --------------------------------------------------------
# Affine
# --------------------------------------------------------
class Affine(Dynamics):
    """
    Affine dynamics
    """

    def __init__(self, state_dyn, input_dyn, affine_dyn):
        super().__init__(state_dyn, input_dyn, affine_dyn)

    # TYPES
    @property
    def is_affine(self):
        return True


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
        self.__dim_per_node = 0

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
    def is_polyhedron_with_identity(self):
        return False

    @property
    def dim_per_node(self):
        """
        :return: size of constraint at each node
        """
        return self.__dim_per_node

    @dim_per_node.setter
    def dim_per_node(self, d):
        self.__dim_per_node = d

    def assign_dual(self, dual, idx, num_vec):
        """
        Assign dual memory for constraints in operator L*
        :param dual: existing memory
        :param idx: dual memory index to start assigning from
        :param num_vec: number of nodes constrained
        :return: assigned memory, index + assigned memory size
        """
        pass

    def op_nonleaf(self, x, n, u):
        """
        Operator: constraint part of L on nonleaf nodes
        :param x: states
        :param n: number of nonleaf nodes
        :param u: inputs
        :return: L(xi, ui)
        """
        pass

    def op_leaf(self, x, n):
        """
        Operator: constraint part of L on leaf nodes
        :param x: leaf states
        :param n: number of leaf nodes
        :return: L(xj)
        """
        pass

    def adj_nonleaf(self, dual, x, n, u):
        """
        Operator: constraint part of L* on nonleaf nodes
        :param x: states
        :param n: number of nonleaf nodes
        :param u: inputs
        :return: L*(xi, ui)
        """
        pass

    def adj_leaf(self, dual, x, n):
        """
        Operator: constraint part of L* on leaf nodes
        :param x: leaf states
        :param n: number of leaf nodes
        :return: L*(xj)
        """
        pass


# --------------------------------------------------------
# None
# --------------------------------------------------------
class No(Constraint):
    """
    No constraint
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
    A rectangle constraint of the form:
    lb <= z <= ub
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
        self.dim_per_node = self.__lb.size

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
            raise Exception("Rectangle constraint - min and max bound dimensions are not equal")
        for i in range(lb.size):
            if lb[i] > ub[i]:
                raise Exception("Rectangle constraint - min greater than max at index (", i, ")")

    def assign_dual(self, dual, idx, num_vec):
        d = [np.array(dual[idx + i * self.dim_per_node:idx + i * self.dim_per_node + self.dim_per_node]).reshape(
            self.dim_per_node, 1) for i in range(num_vec)]
        idx += self.dim_per_node * num_vec
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
    A polyhedral constraint of the form:
    lb <= Gz <= ub
    """

    def __init__(self, matrix, lower_bound, upper_bound):
        """
        :param matrix: pre-multiplying matrix
        :param lower_bound: vector
        :param upper_bound: vector
        """
        super().__init__()
        self.__check_arguments(matrix, lower_bound, upper_bound)
        self.__matrix = matrix
        self.__lb = np.array(lower_bound).reshape(-1, 1)
        self.__ub = np.array(upper_bound).reshape(-1, 1)
        self.dim_per_node = self.__lb.size

    @property
    def is_polyhedron(self):
        return True

    @property
    def matrix(self):
        return self.__matrix

    @property
    def lower_bound(self):
        return self.__lb

    @property
    def upper_bound(self):
        return self.__ub

    @staticmethod
    def __check_arguments(G, lb, ub):
        if not isinstance(G, np.ndarray):
            raise Exception("Polyhedron constraint - matrix is not a numpy array")
        if lb.size != ub.size:
            raise Exception("Polyhedron constraint - min and max bound dimensions are not equal")
        if G.shape[0] != lb.size:
            raise Exception("Polyhedron constraint - matrix and bound dimensions are not equal")
        for i in range(lb.size):
            if lb[i] > ub[i]:
                raise Exception("Polyhedron constraint - min greater than max at index (", i, ")")

    def assign_dual(self, dual, idx, num_vec):
        d = [np.array(dual[idx + i * self.dim_per_node:idx + i * self.dim_per_node + self.dim_per_node]).reshape(
            self.dim_per_node, 1) for i in range(num_vec)]
        idx += self.dim_per_node * num_vec
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


# --------------------------------------------------------
# Polyhedron with identity matrix
# --------------------------------------------------------
class PolyhedronWithIdentity(Constraint):
    """
    A polyhedron constraint of the form:
    lb <= [I (G)']' * z <= ub
    where I is the identity matrix.
    """

    def __init__(self, rect, poly):
        """
        :param rect: instance of class `Rectangle`
        :param poly: instance of class `Polyhedron`
        """
        super().__init__()
        self.__check_arguments(rect, poly)
        self.__rect = rect
        self.__poly = poly

    @staticmethod
    def __check_arguments(rect, poly):
        if not isinstance(rect, Rectangle):
            raise Exception("PolyhedronWithIdentity constraint - not rectangle")
        if not isinstance(poly, Polyhedron):
            raise Exception("PolyhedronWithIdentity constraint - not polyhedron")

    def is_polyhedron_with_identity(self):
        return True

    @property
    def rect(self):
        return self.__rect

    @property
    def poly(self):
        return self.__poly

    def assign_dual(self, dual, idx, num_vec):
        r = self.__rect.dim_per_node
        p = self.__poly.dim_per_node
        d = [np.array(dual[idx + i * r:idx + i * r + r]).reshape(r, 1) for i in range(num_vec)] + [
            np.array(dual[idx + i * p:idx + i * p + p]).reshape(p, 1) for i in range(num_vec)]
        idx += (r + p) * num_vec
        return d, idx

    def op_nonleaf(self, x, n, u):
        dual = [np.zeros((self.__rect.dim_per_node, 1))] * n + [np.zeros((self.__poly.dim_per_node, 1))] * n
        for i in range(n):
            dual[i] = np.vstack((x[i], u[i]))
            dual[i + n] = self.__poly.matrix @ np.vstack((x[i], u[i]))
        return dual

    def op_leaf(self, x, n):
        dual = [np.zeros((self.__rect.dim_per_node, 1))] * n + [np.zeros((self.__poly.dim_per_node, 1))] * n
        for i in range(n):
            dual[i] = x[i]
            dual[i + n] = self.__poly.matrix @ x[i]
        return dual

    def adj_nonleaf(self, dual, x, n, u):
        dim = x[0].size
        for i in range(n):
            x[i] = dual[i][:dim]
            u[i] = dual[i][dim:]
            xu = self.__poly.matrix.T @ dual[i + n]
            x[i] += xu[:dim]
            u[i] += xu[dim:]
        return x, u

    def adj_leaf(self, dual, x, n):
        for i in range(n):
            x[i] = dual[i]
            x[i] += self.__poly.matrix.T @ dual[i + n]
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
