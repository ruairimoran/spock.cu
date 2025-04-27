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

def check_spd(mat, name):
    eigs = np.linalg.eigvals(mat)
    is_positive_definite = eigs.all() > 0
    is_symmetric = np.allclose(mat, mat.T)
    if not (is_positive_definite and is_symmetric):
        raise Exception(f"Invalid cost. The matrix ({name}) is not symmetric positive-definite.")


# --------------------------------------------------------
# Base
# --------------------------------------------------------
class Cost:
    """
    Base class for costs
    """

    def __init__(self, leaf):
        self.nodes = 0
        self.dim_per_node = 0
        self.dim = 0
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

    def assign_dual(self, dual, idx):
        """
        Assign dual memory for costs in operator L*
        :param dual: existing memory
        :param idx: dual memory index to start assigning from
        :return: assigned memory, index + assigned memory size
        """
        pass

    def op_nonleaf(self, x, u, t, ancestors):
        """
        Operator: cost part of L on nonleaf nodes
        :param x: states
        :param u: inputs
        :param t: tau
        :param ancestors: list of ancestors
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

    def adj_nonleaf(self, dual, num_states, num_inputs, ancestors):
        """
        Operator: cost part of L* on nonleaf nodes
        :param dual: dual vector
        :param num_states: number of states
        :param num_inputs: number of inputs
        :param ancestors: list of ancestors
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

    @staticmethod
    def _inv(A, b):
        return np.linalg.solve(A, b)


# --------------------------------------------------------
# Quadratic
# --------------------------------------------------------
class Quadratic(Cost):
    """
    Quadratic costs (internal use)
    """

    def __init__(self, list_of_costs):
        super().__init__(list_of_costs[-1].leaf)
        self.nodes = len(list_of_costs)
        self.__Q = [cost.Q for cost in list_of_costs]
        self.__Q_sqrt = [sqrtm(Q) for Q in self.__Q]
        self.__Q_unconditioned = deepcopy(self.__Q)
        self.dim_per_node = self.__Q[0].shape[0] + 2
        self.__start = 0
        if not self.is_leaf:
            self.__R = [cost.R for cost in list_of_costs]
            self.__R_sqrt = [sqrtm(R) for R in self.__R]
            self.__R_unconditioned = deepcopy(self.__R)
            self.dim_per_node = self.dim_per_node + self.__R[0].shape[0]
            self.__start = 1
        self.dim = self.nodes * self.dim_per_node
        self.__translation = [np.zeros((self.dim_per_node, 1)) for _ in list_of_costs]
        self.__set_translation()

    def __set_translation(self):
        for i in range(self.__start, self.nodes):
            a = np.zeros((self.__Q[i].shape[0], 1))
            c = -.5
            d = .5
            if self.is_leaf:
                self.__translation[i] = np.vstack((a, c, d))
            else:
                b = np.zeros((self.__R[i].shape[0], 1))
                self.__translation[i] = np.vstack((a, b, c, d))

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
        d = [None for _ in range(self.nodes)]
        for i in range(self.nodes):
            d[i] = np.array(dual[idx:idx + self.dim_per_node]).reshape(self.dim_per_node, 1)
            idx += self.dim_per_node
        return d, idx

    def op_nonleaf(self, x, u, t, ancestors):
        d = [np.zeros((self.dim_per_node, 1)) for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            anc = ancestors[i]
            half_t = t[i] * 0.5
            d[i] = np.vstack((self.__Q_sqrt[i] @ x[anc], self.__R_sqrt[i] @ u[anc], half_t, half_t))
        return d

    def op_leaf(self, x, s):
        d = [np.zeros((self.dim_per_node, 1)) for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            half_s = s[i] * 0.5
            d[i] = np.vstack((self.__Q_sqrt[i] @ x[i], half_s, half_s))
        return d

    def adj_nonleaf(self, dual, num_states, num_inputs, ancestors):
        num_si = num_states + num_inputs
        x_ = [np.zeros((num_states, 1)) for _ in range(self.nodes)]
        u_ = [np.zeros((num_inputs, 1)) for _ in range(self.nodes)]
        t_ = [0. for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            dual_i = dual[i]
            anc = ancestors[i]
            x_[anc] += self.__Q_sqrt[i] @ dual_i[:num_states]
            u_[anc] += self.__R_sqrt[i] @ dual_i[num_states:num_si]
            t_[i] = 0.5 * (dual_i[num_si] + dual_i[num_si + 1])
        return x_, u_, t_

    def adj_leaf(self, dual, num_states):
        x_ = [np.zeros((num_states, 1)) for _ in range(self.nodes)]
        s_ = [0. for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            dual_i = dual[i]
            x_[i] = self.__Q_sqrt[i] @ dual_i[:num_states]
            s_[i] = 0.5 * (dual_i[num_states] + dual_i[num_states + 1])
        return x_, s_

    def get_scaling_states(self, scale_x):
        for ele in range(scale_x.size):
            for i in range(self.__start, self.nodes):
                scale = np.sqrt(self.__Q_unconditioned[i][ele, ele])
                if scale > scale_x[ele]:
                    scale_x[ele] = np.asarray(scale).item()
        return scale_x

    def get_scaling_inputs(self, scale_u):
        for ele in range(scale_u.size):
            for i in range(self.__start, self.nodes):
                scale = np.sqrt(self.__R_unconditioned[i][ele, ele])
                if scale > scale_u[ele]:
                    scale_u[ele] = np.asarray(scale).item()
        return scale_u

    def condition(self, scaling_state_inv, scaling_input_inv=None):
        for i in range(self.__start, self.nodes):
            self.__Q[i] = scaling_state_inv.T @ self.Q_uncond[i] @ scaling_state_inv
            self.__Q_sqrt[i] = sqrtm(self.__Q[i])
            if not self.is_leaf:
                self.__R[i] = scaling_input_inv.T @ self.R_uncond[i] @ scaling_input_inv
                self.__R_sqrt[i] = sqrtm(self.__R[i])
        self.__set_translation()
        return self


class CostQuadratic:
    """
    Quadratic costs
    """

    def __init__(self, Q, R=None, leaf=False):
        """
        :param Q: quadratic state cost matrix
        :param R: quadratic input cost matrix
        :param leaf: for leaf nodes
        """
        check_spd(Q, "Q")
        if not leaf:
            check_spd(R, "R")
        self.Q = Q
        self.R = R
        self.leaf = leaf

    def node_zero(self):
        self.Q[:] = 0.
        self.R[:] = 0.

    @staticmethod
    def get_class():
        return Quadratic


# --------------------------------------------------------
# Linear cost
# --------------------------------------------------------
class Linear(Cost):
    """
    Linear costs (internal use)
    """

    def __init__(self, list_of_costs):
        super().__init__(list_of_costs[-1].leaf)
        self.nodes = len(list_of_costs)
        self.__q = [np.array(cost.q).reshape(-1, 1) for cost in list_of_costs]
        self.__q_unconditioned = deepcopy(self.__q)
        self.dim_per_node = self.__q[0].shape[0]
        self.__start = 0
        if not self.is_leaf:
            self.__r = [np.array(cost.r).reshape(-1, 1) for cost in list_of_costs]
            self.__r_unconditioned = deepcopy(self.__r)
            self.dim_per_node = self.dim_per_node + self.__r[0].shape[0]
            self.__start = 1
        self.__grad = [np.zeros((1, self.dim_per_node)) for _ in list_of_costs]
        self.dim = self.nodes * self.dim_per_node
        self.__blocks = 2
        self.__set_grad()

    @property
    def is_linear(self):
        return True

    def __set_grad(self):
        for i in range(self.__start, self.nodes):
            if self.is_leaf:
                self.__grad[i] = self.__q[i].T
            else:
                self.__grad[i] = np.hstack((self.__q[i].T, self.__r[i].T))

    @property
    def cost_gradient(self):
        return self.__grad

    @property
    def q(self):
        return self.__q

    @property
    def q_uncond(self):
        return self.__q_unconditioned

    @property
    def r(self):
        return self.__r

    @property
    def r_uncond(self):
        return self.__r_unconditioned

    def assign_dual(self, dual, idx):
        d = [None, None]
        for i in range(self.__blocks):
            d[i] = np.array(dual[idx:idx + self.nodes]).reshape(self.nodes, 1)
            idx += self.nodes
        return d, idx

    def op_nonleaf(self, x, u, t, ancestors):
        d = [np.zeros((self.nodes, 1)) for _ in range(self.__blocks)]
        for i in range(self.__start, self.nodes):
            anc = ancestors[i]
            d[0][i] = np.array(self.__q[i].T @ x[anc] + self.__r[i].T @ u[anc])
        d[1] = np.array(t).reshape(-1, 1)
        return d

    def op_leaf(self, x, s):
        d = [np.zeros((self.nodes, 1)) for _ in range(self.__blocks)]
        for i in range(self.__start, self.nodes):
            d[0][i] = np.array(self.__q[i].T @ x[i])
        d[1] = np.array(s).reshape(-1, 1)
        return d

    def adj_nonleaf(self, dual, num_states, num_inputs, ancestors):
        x_ = [np.zeros((num_states, 1)) for _ in range(self.nodes)]
        u_ = [np.zeros((num_inputs, 1)) for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            anc = ancestors[i]
            x_[anc] += (self.__q[i] @ dual[0][i]).reshape(-1, 1)
            u_[anc] += (self.__r[i] @ dual[0][i]).reshape(-1, 1)
        t_ = np.array(dual[1]).reshape(-1, 1)
        return x_, u_, t_

    def adj_leaf(self, dual, num_states):
        x_ = [np.zeros((num_states, 1)) for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            x_[i] = (self.__q[i] @ dual[0][i]).reshape(-1, 1)
        s_ = np.array(dual[1]).reshape(-1, 1)
        return x_, s_

    def get_scaling_states(self, scale_x):
        for ele in range(scale_x.size):
            for i in range(self.__start, self.nodes):
                scale = self.__q_unconditioned[i][ele]
                if scale > scale_x[ele]:
                    scale_x[ele] = np.asarray(scale).item()
        return scale_x

    def get_scaling_inputs(self, scale_u):
        for ele in range(scale_u.size):
            for i in range(self.__start, self.nodes):
                scale = self.__r_unconditioned[i][ele]
                if scale > scale_u[ele]:
                    scale_u[ele] = np.asarray(scale).item()
        return scale_u

    def condition(self, scaling_state_inv, scaling_input_inv=None):
        for i in range(self.__start, self.nodes):
            self.__q[i] = np.diagonal(scaling_state_inv).reshape(-1, 1) * self.__q_unconditioned[i]
            if not self.is_leaf:
                self.__r[i] = np.diagonal(scaling_input_inv).reshape(-1, 1) * self.__r_unconditioned[i]
        self.__set_grad()
        return self


class CostLinear:
    """
    Linear costs builder
    """

    def __init__(self, q, r=None, leaf=False):
        """
        :param q: linear state cost vector
        :param r: linear input cost vector
        :param leaf: for leaf nodes
        """
        self.q = q
        self.r = r
        self.leaf = leaf

    def node_zero(self):
        self.q[:] = 0.
        self.r[:] = 0.

    @staticmethod
    def get_class():
        return Linear


# --------------------------------------------------------
# Quadratic-plus-linear cost
# --------------------------------------------------------
class QuadraticPlusLinear(Cost):
    """
    Quadratic-plus-linear costs (internal use)
    """

    def __init__(self, list_of_costs):
        super().__init__(list_of_costs[-1].leaf)
        self.nodes = len(list_of_costs)
        self.__Q = [cost.Q for cost in list_of_costs]
        self.__Q_sqrt = [sqrtm(Q) for Q in self.__Q]
        self.__Q_unconditioned = deepcopy(self.__Q)
        self.__q = [np.array(cost.q).reshape(-1, 1) for cost in list_of_costs]
        self.__q_unconditioned = deepcopy(self.__q)
        self.dim_per_node = self.__Q[0].shape[0] + 2
        self.__start = 0
        if not self.is_leaf:
            self.__R = [cost.R for cost in list_of_costs]
            self.__R_sqrt = [sqrtm(R) for R in self.__R]
            self.__R_unconditioned = deepcopy(self.__R)
            self.__r = [np.array(cost.r).reshape(-1, 1) for cost in list_of_costs]
            self.__r_unconditioned = deepcopy(self.__r)
            self.dim_per_node = self.dim_per_node + self.__R[0].shape[0]
            self.__start = 1
        self.dim = self.nodes * self.dim_per_node
        self.__translation = [np.zeros((self.dim_per_node, 1)) for _ in list_of_costs]
        self.__set_translation()

    def __set_translation(self):
        for i in range(self.__start, self.nodes):
            a = .5 * self._inv(self.__Q_sqrt[i], self.__q[i]).reshape(-1, 1)
            nrm_q = -.125 * self.__q[i].T @ self._inv(self.__Q[i], self.__q[i])
            c = -.5 + nrm_q
            d = .5 + nrm_q
            if self.is_leaf:
                self.__translation[i] = np.vstack((a, c, d))
            else:
                b = .5 * self._inv(self.__R_sqrt[i], self.__r[i]).reshape(-1, 1)
                nrm_r = -.125 * self.__r[i].T @ self._inv(self.__R[i], self.__r[i])
                c = c + nrm_r
                d = d + nrm_r
                self.__translation[i] = np.vstack((a, b, c, d))

    @property
    def is_quadratic_plus_linear(self):
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
    def q(self):
        return self.__q

    @property
    def q_uncond(self):
        return self.__q_unconditioned

    @property
    def r(self):
        return self.__r

    @property
    def r_uncond(self):
        return self.__r_unconditioned

    @property
    def translation(self):
        return self.__translation

    def assign_dual(self, dual, idx):
        d = [None for _ in range(self.nodes)]
        for i in range(self.nodes):
            d[i] = np.array(dual[idx:idx + self.dim_per_node]).reshape(self.dim_per_node, 1)
            idx += self.dim_per_node
        return d, idx

    def op_nonleaf(self, x, u, t, ancestors):
        d = [np.zeros((self.dim_per_node, 1)) for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            anc = ancestors[i]
            half_t = .5 * t[i] - .5 * (self.__q[i].T @ x[anc] + self.__r[i].T @ u[anc])
            d[i] = np.vstack((self.__Q_sqrt[i] @ x[anc], self.__R_sqrt[i] @ u[anc], half_t, half_t))
        return d

    def op_leaf(self, x, s):
        d = [np.zeros((self.dim_per_node, 1)) for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            half_s = .5 * s[i] - .5 * self.__q[i].T @ x[i]
            d[i] = np.vstack((self.__Q_sqrt[i] @ x[i], half_s, half_s))
        return d

    def adj_nonleaf(self, dual, num_states, num_inputs, ancestors):
        num_si = num_states + num_inputs
        x_ = [np.zeros((num_states, 1)) for _ in range(self.nodes)]
        u_ = [np.zeros((num_inputs, 1)) for _ in range(self.nodes)]
        t_ = [0. for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            dual_i = dual[i]
            d0 = dual_i[:num_states]
            d1 = dual_i[num_states:num_si]
            d2 = dual_i[num_si].reshape(1, 1)
            d3 = dual_i[num_si + 1].reshape(1, 1)
            anc = ancestors[i]
            x_[anc] += self.__Q_sqrt[i] @ d0 - .5 * self.__q[i] @ d2 - .5 * self.__q[i] @ d3
            u_[anc] += self.__R_sqrt[i] @ d1 - .5 * self.__r[i] @ d2 - .5 * self.__r[i] @ d3
            t_[i] = 0.5 * (d2 + d3)
        return x_, u_, t_

    def adj_leaf(self, dual, num_states):
        x_ = [np.zeros((num_states, 1)) for _ in range(self.nodes)]
        s_ = [0. for _ in range(self.nodes)]
        for i in range(self.__start, self.nodes):
            dual_i = dual[i]
            d0 = dual_i[:num_states]
            d2 = dual_i[num_states].reshape(1, 1)
            d3 = dual_i[num_states + 1].reshape(1, 1)
            x_[i] = self.__Q_sqrt[i] @ d0 - .5 * self.__q[i] @ d2 - .5 * self.__q[i] @ d3
            s_[i] = 0.5 * (d2 + d3)
        return x_, s_

    def get_scaling_states(self, scale_x):
        for ele in range(scale_x.size):
            for i in range(self.__start, self.nodes):
                scale = np.sqrt(self.__Q_unconditioned[i][ele, ele])
                if scale > scale_x[ele]:
                    scale_x[ele] = np.asarray(scale).item()
                scale = np.sqrt(abs(self.__q_unconditioned[i][ele]))
                if scale > scale_x[ele]:
                    scale_x[ele] = np.asarray(scale).item()
        return scale_x

    def get_scaling_inputs(self, scale_u):
        for ele in range(scale_u.size):
            for i in range(self.__start, self.nodes):
                scale = np.sqrt(self.__R_unconditioned[i][ele, ele])
                if scale > scale_u[ele]:
                    scale_u[ele] = np.asarray(scale).item()
                scale = np.sqrt(abs(self.__r_unconditioned[i][ele]))
                if scale > scale_u[ele]:
                    scale_u[ele] = np.asarray(scale).item()
        return scale_u

    def condition(self, scaling_state_inv, scaling_input_inv=None):
        for i in range(self.__start, self.nodes):
            self.__Q[i] = scaling_state_inv.T @ self.Q_uncond[i] @ scaling_state_inv
            self.__Q_sqrt[i] = sqrtm(self.__Q[i])
            self.__q[i] = np.diagonal(scaling_state_inv).reshape(-1, 1) * self.__q_unconditioned[i]
            if not self.is_leaf:
                self.__R[i] = scaling_input_inv.T @ self.R_uncond[i] @ scaling_input_inv
                self.__R_sqrt[i] = sqrtm(self.__R[i])
                self.__r[i] = np.diagonal(scaling_input_inv).reshape(-1, 1) * self.__r_unconditioned[i]
        self.__set_translation()
        return self


class CostQuadraticPlusLinear:
    """
    Quadratic-plus-linear costs
    """

    def __init__(self, Q, q, R=None, r=None, leaf=False):
        """
        :param Q: quadratic state cost matrix
        :param q: linear state cost vector
        :param R: quadratic input cost matrix
        :param r: linear input cost vector
        :param leaf: for leaf nodes
        """
        check_spd(Q, "Q")
        if not leaf:
            check_spd(R, "R")
        self.Q = Q
        self.R = R
        self.q = q
        self.r = r
        self.leaf = leaf

    def node_zero(self):
        self.Q[:] = 0.
        self.R[:] = 0.
        self.q[:] = 0.
        self.r[:] = 0.

    @staticmethod
    def get_class():
        return QuadraticPlusLinear


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
