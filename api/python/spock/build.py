import numpy as np
from copy import deepcopy
from scipy.linalg import sqrtm


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

    def __init__(self, state_, input_, constant_=None):
        self.__is_affine = constant_ is not None
        self.__state = state_
        self.__input = input_
        self.__const = np.array(constant_).reshape(-1, 1) if self.__is_affine else np.zeros((state_.shape[0], 1))
        self.__state_input = np.hstack((self.__state, self.__input))
        self.__state_unconditioned = deepcopy(self.__state)
        self.__input_unconditioned = deepcopy(self.__input)
        self.__const_unconditioned = deepcopy(self.__const)
        self.__state_input_unconditioned = np.hstack((self.__state_unconditioned, self.__input_unconditioned))

    # TYPES
    @property
    def is_linear(self):
        return not self.__is_affine

    @property
    def is_affine(self):
        return self.__is_affine

    @property
    def A(self):
        return self.__state

    @property
    def B(self):
        return self.__input

    @property
    def c(self):
        return self.__const

    @property
    def A_B(self):
        return self.__state_input

    @property
    def A_uncond(self):
        return self.__state_unconditioned

    @property
    def B_uncond(self):
        return self.__input_unconditioned

    @property
    def c_uncond(self):
        return self.__const_unconditioned

    @property
    def A_B_uncond(self):
        return self.__state_input_unconditioned

    def condition(self, scale_x_inv, scale_u_inv, scale_x):
        self.__state = scale_x @ self.__state_unconditioned @ scale_x_inv
        self.__input = scale_x @ self.__input_unconditioned @ scale_u_inv
        self.__const = scale_x @ self.__const_unconditioned
        self.__state_input = np.hstack((self.__state, self.__input))
        return self


# =====================================================================================================================
# Costs
# =====================================================================================================================

# --------------------------------------------------------
# Base
# --------------------------------------------------------
class Cost:
    """
    Base class for costs
    """

    def __init__(self, leaf):
        self.__dim_per_node = 0
        self.__leaf = leaf

    # TYPES
    @property
    def is_leaf(self):
        return self.__leaf

    @property
    def is_quadratic(self):
        return False

    @property
    def is_linear(self):
        return False

    @property
    def is_quadratic_plus_linear(self):
        return False

    @property
    def dim_per_node(self):
        """
        :return: size of cost at each node
        """
        return self.__dim_per_node

    @dim_per_node.setter
    def dim_per_node(self, d):
        self.__dim_per_node = d

    def assign_dual(self, dual, idx):
        """
        Assign dual memory for costs in operator L*
        :param dual: existing memory
        :param idx: dual memory index to start assigning from
        :return: assigned memory, index + assigned memory size
        """
        pass

    def op_nonleaf(self, x_anc, u_anc, t):
        """
        Operator: cost part of L on nonleaf nodes
        :param x_anc: ancestor state
        :param u_anc: ancestor input
        :param t: tau
        :return: L(xi, ui, ti)
        """
        pass

    def op_leaf(self, x, s):
        """
        Operator: cost part of L on leaf nodes
        :param x: leaf state
        :param s: s
        :return: L(xj, sj)
        """
        pass

    def adj_nonleaf(self, dual, num_states, num_inputs):
        """
        Operator: cost part of L* on nonleaf nodes
        :param dual: dual vector
        :param num_states: number of states
        :param num_inputs: number of inputs
        :return: L*(xi, ui, ti)
        """
        pass

    def adj_leaf(self, dual, num_states):
        """
        Operator: cost part of L* on leaf nodes
        :param dual: dual vector
        :param num_states: number of states
        :return: L*(xj, sj)
        """
        pass

    def condition(self, scale_x, scale_u=None):
        """
        Scale cost
        """
        return self

    def node_zero(self):
        """
        Set to zero
        """
        return self

    @staticmethod
    def _inv(A, b):
        return np.linalg.solve(A, b)


# --------------------------------------------------------
# Quadratic
# --------------------------------------------------------
class Quadratic(Cost):
    """
    Quadratic costs
    """

    def __init__(self, Q, R=None, leaf=False):
        """
        :param Q: quadratic state cost matrix
        :param R: quadratic input cost matrix
        :param leaf: for leaf nodes
        """
        super().__init__(leaf)
        self.__Q = Q
        self.__Q_sqrt = sqrtm(self.__Q)
        self.__Q_unconditioned = deepcopy(self.__Q)
        r_ = R is not None
        self.__R = R if r_ else None
        self.__R_sqrt = sqrtm(self.__R) if r_ else None
        self.__R_unconditioned = deepcopy(self.__R)
        self.__translation = None
        self.__node_zero = False
        self.dim_per_node = self.Q.shape[0] + self.R.shape[0] + 2 if not leaf else self.Q.shape[0] + 2
        self.__set_translation()

    def __set_translation(self, zero=False):
        a = np.zeros((self.Q.shape[0], 1))
        if zero:
            c = 0
            d = 0
        else:
            c = -.5
            d = .5
        if self.is_leaf:
            self.__translation = np.vstack((a, c, d))
        else:
            b = np.zeros((self.R.shape[0], 1))
            self.__translation = np.vstack((a, b, c, d))

    @property
    def is_quadratic(self):
        return True

    @property
    def Q(self):
        return self.__Q

    @property
    def Q_sqrt(self):
        return self.__Q_sqrt

    @property
    def Q_uncond(self):
        return self.__Q_unconditioned

    @property
    def R(self):
        return self.__R

    @property
    def R_sqrt(self):
        return self.__R_sqrt

    @property
    def R_uncond(self):
        return self.__R_unconditioned

    @property
    def translation(self):
        return self.__translation

    def assign_dual(self, dual, idx):
        d = np.array(dual[idx:idx + self.dim_per_node]).reshape(self.dim_per_node, 1)
        idx += self.dim_per_node
        return d, idx

    def op_nonleaf(self, x_anc, u_anc, t):
        if self.__node_zero:
            return np.zeros((self.dim_per_node, 1))
        else:
            half_t = t * 0.5
            return np.vstack((self.__Q_sqrt @ x_anc, self.__R_sqrt @ u_anc, half_t, half_t))

    def op_leaf(self, x, s):
        if self.__node_zero:
            return np.zeros((self.dim_per_node, 1))
        else:
            half_s = s * 0.5
            return np.vstack((self.__Q_sqrt @ x, half_s, half_s))

    def adj_nonleaf(self, dual, num_states, num_inputs):
        num_si = num_states + num_inputs
        x_ = self.__Q_sqrt @ dual[:num_states]
        u_ = self.__R_sqrt @ dual[num_states:num_si]
        t_ = 0.5 * (dual[num_si] + dual[num_si + 1])
        return x_, u_, t_

    def adj_leaf(self, dual, num_states):
        x_ = self.__Q_sqrt @ dual[:num_states]
        s_ = 0.5 * (dual[num_states] + dual[num_states + 1])
        return x_, s_

    def condition(self, scaling_state_inv, scaling_input_inv=None):
        if not self.__node_zero:
            self.__Q = scaling_state_inv.T @ self.Q_uncond @ scaling_state_inv
            self.__R = scaling_input_inv.T @ self.R_uncond @ scaling_input_inv if not self.is_leaf else None
            self.__set_translation()
        return self

    def node_zero(self):
        self.__node_zero = True
        self.__set_translation(True)
        return self


# --------------------------------------------------------
# Linear cost
# --------------------------------------------------------
class Linear(Cost):
    """
    Linear costs
    """

    def __init__(self, q, r=None, leaf=False):
        """
        :param q: linear state cost vector
        :param r: linear input cost vector
        :param leaf: for leaf nodes
        """
        super().__init__(leaf)
        self.__q = np.array(q).reshape(-1, 1)
        self.__q_unconditioned = deepcopy(self.__q)
        self.__r = np.array(r).reshape(-1, 1)
        self.__r_unconditioned = deepcopy(self.__r)
        self.__gradient = None
        self.__node_zero = False
        self.dim_per_node = 2
        self.__set_gradient()

    @property
    def is_linear(self):
        return True

    def __set_gradient(self):
        if self.is_leaf:
            self.__gradient = self.__q
        else:
            self.__gradient = np.vstack((self.__q, self.__r))

    @property
    def gradient(self):
        return self.__gradient

    @property
    def q_uncond(self):
        return self.__q_unconditioned

    @property
    def r_uncond(self):
        return self.__r_unconditioned

    def assign_dual(self, dual, idx):
        d = np.array(dual[idx:idx + self.dim_per_node]).reshape(self.dim_per_node, 1)
        idx += self.dim_per_node
        return d, idx

    def op_nonleaf(self, x_anc, u_anc, t):
        if self.__node_zero:
            return np.zeros((self.dim_per_node, 1))
        else:
            return np.vstack((self.__q.T @ x_anc + self.__r.T @ u_anc, t))

    def op_leaf(self, x, s):
        if self.__node_zero:
            return np.zeros((self.dim_per_node, 1))
        else:
            return np.vstack((self.__q.T @ x, s))

    def adj_nonleaf(self, dual, num_states, num_inputs):
        x_ = (self.__q @ dual[0]).reshape(-1, 1)
        u_ = (self.__r @ dual[0]).reshape(-1, 1)
        t_ = dual[1][0]
        return x_, u_, t_

    def adj_leaf(self, dual, num_states):
        x_ = (self.__q @ dual[0]).reshape(-1, 1)
        s_ = dual[1][0]
        return x_, s_

    def condition(self, scaling_state_inv, scaling_input_inv=None):
        if not self.__node_zero:
            self.__q = np.diagonal(scaling_state_inv).reshape(-1, 1) * self.q_uncond
            self.__r = np.diagonal(scaling_input_inv).reshape(-1, 1) * self.r_uncond if not self.is_leaf else None
            self.__set_gradient()
        return self

    def node_zero(self):
        self.__node_zero = True
        self.__q = np.zeros(self.q_uncond.shape)
        self.__r = np.zeros(self.r_uncond.shape) if not self.is_leaf else None
        self.__set_gradient()
        return self


# --------------------------------------------------------
# Quadratic-plus-linear cost
# --------------------------------------------------------
class QuadraticPlusLinear(Cost):
    """
    Quadratic and linear costs
    """

    def __init__(self, Q, R=None, q=None, r=None, leaf=False):
        """
        :param Q: quadratic state cost matrix
        :param R: quadratic input cost matrix
        :param q: linear state cost vector
        :param r: linear input cost vector
        :param leaf: for leaf nodes
        """
        super().__init__(leaf)
        self.__Q = Q
        self.__Q_sqrt = sqrtm(self.__Q)
        self.__Q_unconditioned = deepcopy(self.__Q)
        self.__R = R
        self.__R_sqrt = sqrtm(self.__R)
        self.__R_unconditioned = deepcopy(self.__R)
        self.__q = np.array(q).reshape(-1, 1)
        self.__q_unconditioned = deepcopy(self.__q)
        self.__r = np.array(r).reshape(-1, 1)
        self.__r_unconditioned = deepcopy(self.__r)
        self.__node_zero = False
        self.dim_per_node = self.Q.shape[0] + self.R.shape[0] + 2 if not leaf else self.Q.shape[0] + 2
        self.__set_translation()

    @property
    def is_quadratic_plus_linear(self):
        return True

    def __set_translation(self, zero=False):
        if zero:
            a = np.zeros((self.Q.shape[0], 1))
            b = np.zeros((self.R.shape[0], 1))
            c = np.zeros((1, 1))
            d = np.zeros((1, 1))
        else:
            nrm_q = self.__q.T @ self._inv(self.Q, self.__q)
            nrm_r = self.__r.T @ self._inv(self.R, self.__r)
            scaled_nrm = .125 * (nrm_q + nrm_r)
            a = self._inv(self.Q_sqrt, self.__q).reshape(-1, 1)
            b = self._inv(self.__R_sqrt, self.__r).reshape(-1, 1) if not self.is_leaf else None
            c = -.5 + scaled_nrm
            d = .5 + scaled_nrm
        if self.is_leaf:
            self.__translation = np.vstack((a, c, d))
        else:
            self.__translation = np.vstack((a, b, c, d))

    @property
    def Q(self):
        return self.__Q

    @property
    def Q_sqrt(self):
        return self.__Q_sqrt

    @property
    def Q_uncond(self):
        return self.__Q_unconditioned

    @property
    def R(self):
        return self.__R

    @property
    def R_sqrt(self):
        return self.__R_sqrt

    @property
    def R_uncond(self):
        return self.__R_unconditioned

    @property
    def q_uncond(self):
        return self.__q_unconditioned

    @property
    def r_uncond(self):
        return self.__r_unconditioned

    def assign_dual(self, dual, idx):
        d = np.array(dual[idx:idx + self.dim_per_node]).reshape(self.dim_per_node, 1)
        idx += self.dim_per_node
        return d, idx

    def op_nonleaf(self, x_anc, u_anc, t):
        pass

    def op_leaf(self, x, s):
        pass

    def adj_nonleaf(self, dual, num_states, num_inputs):
        pass

    def adj_leaf(self, dual, num_states):
        pass

    def condition(self, scaling_state_inv, scaling_input_inv=None):
        if not self.__node_zero:
            self.__Q = scaling_state_inv.T @ self.Q_uncond @ scaling_state_inv
            self.__R = scaling_input_inv.T @ self.R_uncond @ scaling_input_inv if not self.is_leaf else None
            self.__q = scaling_state_inv.T @ self.q_uncond
            self.__r = scaling_input_inv.T @ self.r_uncond if not self.is_leaf else None
            self.__set_translation()
        return self

    def node_zero(self):
        self.__node_zero = True
        self.__set_translation(True)
        return self


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

    def condition(self, scale_xu):
        """
        Scale constraint bounds or gamma matrix.
        """
        return self


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
        self.__lo_bound = np.array(lower_bound).reshape(-1, 1)
        self.__up_bound = np.array(upper_bound).reshape(-1, 1)
        self.__lo_bound_unconditioned = deepcopy(self.__lo_bound)
        self.__up_bound_unconditioned = deepcopy(self.__up_bound)
        self.dim_per_node = self.__lo_bound.size

    @property
    def is_rectangle(self):
        return True

    @property
    def lower_bound(self):
        return self.__lo_bound

    @property
    def upper_bound(self):
        return self.__up_bound

    @property
    def lower_bound_uncond(self):
        return self.__lo_bound_unconditioned

    @property
    def upper_bound_uncond(self):
        return self.__up_bound_unconditioned

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

    def condition(self, scale_xu):
        self.__lo_bound = scale_xu @ self.__lo_bound_unconditioned
        self.__up_bound = scale_xu @ self.__up_bound_unconditioned
        return self


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
        self.__lo_bound = np.array(lower_bound).reshape(-1, 1)
        self.__up_bound = np.array(upper_bound).reshape(-1, 1)
        self.__matrix_unconditioned = deepcopy(matrix)
        self.__lo_bound_unconditioned = deepcopy(self.__lo_bound)
        self.__up_bound_unconditioned = deepcopy(self.__up_bound)
        self.dim_per_node = self.__lo_bound.size

    @property
    def is_polyhedron(self):
        return True

    @property
    def matrix(self):
        return self.__matrix

    @property
    def lower_bound(self):
        return self.__lo_bound

    @property
    def upper_bound(self):
        return self.__up_bound

    @property
    def matrix_uncond(self):
        return self.__matrix_unconditioned

    @property
    def lower_bound_uncond(self):
        return self.__lo_bound_unconditioned

    @property
    def upper_bound_uncond(self):
        return self.__up_bound_unconditioned

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

    def condition(self, scale_xu):
        scale_inv = np.diag(1 / np.diagonal(scale_xu))
        self.__matrix = self.__matrix_unconditioned @ scale_inv
        return self


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

    def condition(self, scale_xu):
        self.__rect.condition(scale_xu)
        self.__poly.condition(scale_xu)
        return self


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
