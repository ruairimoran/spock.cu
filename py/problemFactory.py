import jinja2 as j2
import os
import numpy as np
import scipy as sp
from scipy.linalg import sqrtm
from scipy.sparse.linalg import LinearOperator, eigs
import gputils_api as ga
import cvxpy as cvx
from copy import deepcopy
from . import treeFactory
from . import build


class Problem:
    """
    Risk-averse optimal control problem and offline cache storage
    """

    def __init__(self, scenario_tree: treeFactory.Tree, num_states, num_inputs, dynamics,
                 state_cost, input_cost, terminal_cost, nonleaf_constraint, leaf_constraint, risk, test):
        """
        :param scenario_tree: instance of ScenarioTree
        :param num_states: number of system states
        :param num_inputs: number of system inputs
        :param dynamics: list of dynamics (size: num_nodes, used: 1 to num_nodes)
        :param state_cost: list of state cost matrices (size: num_nodes, used: 1 to num_nodes)
        :param input_cost: list of input cost matrices (size: num_nodes, used: 1 to num_nodes)
        :param terminal_cost: list of terminal cost matrices (size: num_nodes, used: num_nonleaf_nodes to num_nodes)
        :param nonleaf_constraint: state-input constraint class (size: 1)
        :param leaf_constraint: state constraint class (size: 1)
        :param risk: list of risk classes (size: num_nonleaf_nodes, used: all)
        :param test: whether to compute test data

        Note: avoid using this constructor directly; use a factory instead
        """
        self.__tree = scenario_tree
        self.__num_states = num_states
        self.__num_inputs = num_inputs
        self.__list_of_dynamics = dynamics
        self.__list_of_nonleaf_state_costs = state_cost
        self.__list_of_nonleaf_input_costs = input_cost
        self.__list_of_leaf_state_costs = terminal_cost
        self.__list_of_risks = risk
        self.__nonleaf_constraint = nonleaf_constraint
        self.__leaf_constraint = leaf_constraint
        self.__test = test
        # Dynamics projection
        self.__P = [np.zeros((self.__num_states, self.__num_states))] * self.__tree.num_nodes
        self.__q = [np.zeros((self.__num_states, 1))] * self.__tree.num_nodes
        self.__K = [np.zeros((self.__num_states, self.__num_states))] * self.__tree.num_nonleaf_nodes
        self.__d = [np.zeros((self.__num_states, 1))] * self.__tree.num_nonleaf_nodes
        self.__lower = True  # Do not change!
        self.__cholesky_lower = [None] * self.__tree.num_nonleaf_nodes
        self.__sum_of_dynamics_tr = [np.zeros((num_states, num_states))] * self.__tree.num_nodes  # A+B@K
        self.__At_P_B = [np.zeros((num_states, num_inputs))] * self.__tree.num_nodes  # At@P@B
        self.__dp_test_init_state = []
        self.__dp_test_states = []
        self.__dp_test_inputs = []
        self.__dp_projected_states = []
        self.__dp_projected_inputs = []
        # Kernel projection
        self.__kernel_constraint_matrix = [np.zeros((0, 0))] * self.__tree.num_nonleaf_nodes
        self.__nullspace_projection_matrix = [np.zeros((0, 0))] * self.__tree.num_nonleaf_nodes
        self.__kernel_constraint_matrix_rows = []
        self.__max_nullspace_dim = self.__tree.num_events * 4 + 1
        self.__projected_ns = [np.zeros((0, 0))] * self.__tree.num_nonleaf_nodes
        # L operator and adjoint testing
        self.__prim_before_op = []
        self.__dual_after_op_before_adj = []
        self.__prim_after_adj = []
        self.__dot_vector = []
        self.__dot_result = []
        # Other
        self.__y_size = 2 * self.__tree.num_events + 1
        self.__step_size = 0
        self.__padded_b = [np.zeros((0, 0))] * self.__tree.num_nonleaf_nodes
        self.__sqrt_nonleaf_state_costs = [np.zeros((0, 0))] * self.__tree.num_nodes
        self.__sqrt_nonleaf_input_costs = [np.zeros((0, 0))] * self.__tree.num_nodes
        self.__sqrt_leaf_state_costs = [np.zeros((0, 0))] * self.__tree.num_leaf_nodes
        # Generate data and files
        print("Computing offline data...")
        self.__generate_offline()
        print("Generating problem files...")
        self.__generate_problem_files()
        # Print problem
        self.__print()

    # GETTERS
    def dynamics_at_node(self, idx):
        return self.__list_of_dynamics[idx]

    def nonleaf_state_cost_at_node(self, idx):
        return self.__list_of_nonleaf_state_costs[idx]

    def nonleaf_input_cost_at_node(self, idx):
        return self.__list_of_nonleaf_input_costs[idx]

    def leaf_state_cost_at_node(self, idx):
        return self.__list_of_leaf_state_costs[idx]

    def nonleaf_constraint(self):
        return self.__nonleaf_constraint

    def leaf_constraint(self):
        return self.__leaf_constraint

    def risk_at_node(self, idx):
        return self.__list_of_risks[idx]

    def __generate_problem_files(self):
        # Setup jinja environment
        file_loader = j2.FileSystemLoader(searchpath=["py/"])
        env = j2.Environment(loader=file_loader,
                             trim_blocks=True,
                             lstrip_blocks=True,
                             block_start_string="\% ",
                             block_end_string="%\\",
                             variable_start_string="\~",
                             variable_end_string="~\\",
                             comment_start_string="\#",
                             comment_end_string="#\\")
        # Generate "problemData.json" from template "problemTemplate.json.jinja2"
        template = env.get_template("data.json.jinja2")
        output = template.render(num_events=self.__tree.num_events,
                                 num_nonleaf_nodes=self.__tree.num_nonleaf_nodes,
                                 num_nodes=self.__tree.num_nodes,
                                 num_stages=self.__tree.num_stages,
                                 num_states=self.__num_states,
                                 num_inputs=self.__num_inputs,
                                 dynamics=self.__list_of_dynamics[0],
                                 nonleaf_constraint=self.__nonleaf_constraint,
                                 leaf_constraint=self.__leaf_constraint,
                                 risk=self.__list_of_risks[0],
                                 ker_con_rows=self.__kernel_constraint_matrix_rows,
                                 null_dim=self.__max_nullspace_dim,
                                 step_size=self.__step_size
                                 )
        path = os.path.join(os.getcwd(), self.__tree.folder)
        os.makedirs(path, exist_ok=True)
        output_file = os.path.join(path, "data.json")
        fh = open(output_file, "w")
        fh.write(output)
        fh.close()
        # Generate stacks
        stack_input_dyn_tr = np.dstack([a.input.T for a in self.__list_of_dynamics])
        stack_AB_dyn = np.dstack([a.state_input for a in self.__list_of_dynamics])
        stack_sqrt_state_cost = np.dstack(self.__sqrt_nonleaf_state_costs)
        stack_sqrt_input_cost = np.dstack(self.__sqrt_nonleaf_input_costs)
        stack_sqrt_terminal_cost = np.dstack(self.__sqrt_leaf_state_costs)
        stack_P = np.dstack(self.__P)
        stack_K = np.dstack(self.__K)
        stack_low_chol = np.dstack(self.__cholesky_lower)
        stack_dyn_tr = np.dstack(self.__sum_of_dynamics_tr)
        stack_APB = np.dstack(self.__At_P_B)
        stack_null = np.dstack(self.__nullspace_projection_matrix)
        stack_b = np.dstack(self.__padded_b)
        # Create tensor dict
        tensors = {
            "inputDynTr": stack_input_dyn_tr,
            "AB_dyn": stack_AB_dyn,
            "sqrtStateCost": stack_sqrt_state_cost,
            "sqrtInputCost": stack_sqrt_input_cost,
            "sqrtTerminalCost": stack_sqrt_terminal_cost,
            "P": stack_P,
            "K": stack_K,
            "lowChol": stack_low_chol,
            "dynTr": stack_dyn_tr,
            "APB": stack_APB,
            "NNtr": stack_null,
            "b": stack_b
        }
        if self.__list_of_dynamics[0].is_affine:
            stack_affine_dyn = np.dstack([dyn.affine for dyn in self.__list_of_dynamics])
            tensors.update({
                "affine_dyn": stack_affine_dyn
            })
        l_con = [self.__nonleaf_constraint, self.__leaf_constraint]
        l_txt = ["nonleafConstraint", "leafConstraint"]
        for i in range(len(l_con)):
            con = l_con[i]
            txt = l_txt[i]
            if con.is_rectangle:
                tensors.update({
                    txt + "ILB": con.lower_bound,
                    txt + "IUB": con.upper_bound
                })
            if con.is_polyhedron:
                tensors.update({
                    txt + "Gamma": con.matrix,
                    txt + "GLB": con.lower_bound,
                    txt + "GUB": con.upper_bound
                })
            if con.is_polyhedron_with_identity:
                tensors.update({
                    txt + "ILB": con.rect.lower_bound,
                    txt + "IUB": con.rect.upper_bound,
                    txt + "Gamma": con.poly.matrix,
                    txt + "GLB": con.poly.lower_bound,
                    txt + "GUB": con.poly.upper_bound
                })
        # Generate files
        for name, tensor in tensors.items():
            self.__tree.write_to_file_fp(name, tensor)
        if self.__test:
            stack_ker_con = np.dstack(self.__kernel_constraint_matrix)
            test_tensors = {
                "S2": stack_ker_con
            }
            test_vectors = {
                "dpTestStates": self.__dp_test_states.reshape(-1),
                "dpTestInputs": self.__dp_test_inputs.reshape(-1),
                "dpProjectedStates": self.__dp_projected_states.reshape(-1),
                "dpProjectedInputs": self.__dp_projected_inputs.reshape(-1),
                "primBeforeOp": np.array(self.__prim_before_op),
                "dualAfterOpBeforeAdj": np.array(self.__dual_after_op_before_adj),
                "primAfterAdj": np.array(self.__prim_after_adj),
                "dotVector": np.array(self.__dot_vector),
                "dotResult": np.array([self.__dot_result])
            }
            for name, tensor in test_tensors.items():
                self.__tree.write_to_file_fp(name, tensor)
            for name, vector in test_vectors.items():
                self.__tree.write_to_file_fp(name, vector)

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    def __generate_offline(self):
        self.__offline_projection_dynamics()
        self.__offline_projection_kernel()
        self.__pad_b()
        self.__sqrt_costs()
        self.__get_step_size()
        if self.__test:
            self.__test_dynamic_programming()
            self.__test_op_and_adj()
            self.__test_dot()  # Must be after `get_step_size()`

    def __offline_projection_dynamics(self):
        for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            self.__P[i] = np.eye(self.__num_states)
        for i in reversed(range(self.__tree.num_nonleaf_nodes)):
            children_of_i = self.__tree.children_of_node(i)
            sum_for_r = 0
            sum_for_k = 0
            for j in children_of_i:
                sum_for_r = sum_for_r \
                            + self.__list_of_dynamics[j].input.T @ self.__P[j] @ self.__list_of_dynamics[j].input
                sum_for_k = sum_for_k \
                            + self.__list_of_dynamics[j].input.T @ self.__P[j] @ self.__list_of_dynamics[j].state
            r_tilde = np.eye(self.__num_inputs) + sum_for_r
            self.__cholesky_lower[i] = sp.linalg.cholesky(r_tilde, lower=self.__lower, check_finite=True)
            self.__K[i] = sp.linalg.cho_solve((self.__cholesky_lower[i], self.__lower), -sum_for_k)
            sum_for_p = 0
            for j in children_of_i:
                sum_of_dynamics = self.__list_of_dynamics[j].state + self.__list_of_dynamics[j].input @ self.__K[i]
                self.__sum_of_dynamics_tr[j] = sum_of_dynamics.T
                sum_for_p = sum_for_p + self.__sum_of_dynamics_tr[j] @ self.__P[j] @ sum_of_dynamics
                self.__At_P_B[j] = self.__sum_of_dynamics_tr[j] @ self.__P[j] @ self.__list_of_dynamics[j].input
            self.__P[i] = np.eye(self.__num_states) + self.__K[i].T @ self.__K[i] + sum_for_p

    def __offline_projection_kernel(self):
        for i in range(self.__tree.num_nonleaf_nodes):
            num_ch = len(self.__tree.children_of_node(i))
            fill = self.__tree.num_events - num_ch
            e_tr_pad = np.pad(self.__list_of_risks[i].e.T, [(0, fill), (0, fill * 2)], mode="constant",
                              constant_values=0.)
            eye = np.eye(num_ch)
            eye_pad = np.pad(eye, [(0, fill), (0, fill)], mode="constant", constant_values=0.)
            if self.__list_of_risks[i].is_avar:
                row1 = np.hstack((e_tr_pad, -eye_pad, -eye_pad))
                row2 = np.zeros((0, 4 * self.__tree.num_events + 1))
            else:
                raise ValueError("Risk type not supported")
            self.__kernel_constraint_matrix[i] = np.vstack((row1, row2))
            n = sp.linalg.null_space(self.__kernel_constraint_matrix[i])
            self.__nullspace_projection_matrix[i] = n @ n.T
            self.__kernel_constraint_matrix_rows = self.__kernel_constraint_matrix[i].shape[0]

    def __test_dynamic_programming(self):
        # Solve with cvxpy
        x_bar = 10 * np.random.randn(self.__num_states, self.__tree.num_nodes)
        u_bar = 10 * np.random.randn(self.__num_inputs, self.__tree.num_nonleaf_nodes)
        x = cvx.Variable((self.__num_states, self.__tree.num_nodes))
        u = cvx.Variable((self.__num_inputs, self.__tree.num_nonleaf_nodes))
        self.__dp_test_init_state = x_bar[:, 0]
        # Sum problem objectives and concatenate constraints
        cost = 0
        constraints = [x[:, 0] == x_bar[:, 0]]
        # Nonleaf nodes
        for node in range(self.__tree.num_nonleaf_nodes):
            cost += cvx.sum_squares(x[:, node] - x_bar[:, node]) + cvx.sum_squares(u[:, node] - u_bar[:, node])
            for ch in self.__tree.children_of_node(node):
                constraints += [x[:, ch] ==
                                self.__list_of_dynamics[ch].state @ x[:, node] +
                                self.__list_of_dynamics[ch].input @ u[:, node] +
                                self.__list_of_dynamics[ch].affine.reshape(-1)]  # affine=zeros if linear dynamics

        # Leaf nodes
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            cost += cvx.sum_squares(x[:, node] - x_bar[:, node])

        problem = cvx.Problem(cvx.Minimize(cost), constraints)
        problem.solve()
        self.__dp_test_states = x_bar.T
        self.__dp_test_inputs = u_bar.T
        self.__dp_projected_states = x.value.T
        self.__dp_projected_inputs = u.value.T

    def __pad_b(self):
        for i in range(self.__tree.num_nonleaf_nodes):
            pad = 2 * (self.__tree.num_events - len(self.__tree.children_of_node(i)))
            self.__padded_b[i] = np.pad(self.__list_of_risks[i].b, [(0, pad), (0, 0)], mode="constant",
                                        constant_values=0.)

    def __sqrt_costs(self):
        for i in range(self.__tree.num_nodes):
            self.__sqrt_nonleaf_state_costs[i] = sqrtm(self.__list_of_nonleaf_state_costs[i])
            self.__sqrt_nonleaf_input_costs[i] = sqrtm(self.__list_of_nonleaf_input_costs[i])

        for i in range(self.__tree.num_leaf_nodes):
            self.__sqrt_leaf_state_costs[i] = sqrtm(self.__list_of_leaf_state_costs[i])

    @staticmethod
    def __flatten(list_of_vectors):
        return np.hstack(np.vstack(list_of_vectors)).tolist()

    def __op(self, prim):
        # Split prim into u, x, y, t, s
        idx = 0
        u = [np.array(prim[idx + i * self.__num_inputs:idx + i * self.__num_inputs + self.__num_inputs]).reshape(
            self.__num_inputs, 1) for i in range(self.__tree.num_nonleaf_nodes)]
        idx += self.__num_inputs * self.__tree.num_nonleaf_nodes
        x = [np.array(prim[idx + i * self.__num_states:idx + i * self.__num_states + self.__num_states]).reshape(
            self.__num_states, 1) for i in range(self.__tree.num_nodes)]
        idx += self.__num_states * self.__tree.num_nodes
        self.__y_size = 2 * self.__tree.num_events + 1
        y = [np.array(prim[idx + i * self.__y_size:idx + i * self.__y_size + self.__y_size]).reshape(
            self.__y_size, 1) for i in range(self.__tree.num_nonleaf_nodes)]
        idx += self.__y_size * self.__tree.num_nonleaf_nodes
        t = prim[idx:idx + self.__tree.num_nodes]
        idx += self.__tree.num_nodes
        s = prim[idx:]
        num_si = self.__num_states + self.__num_inputs

        # -> i
        i_ = deepcopy(y)

        # -> ii
        ii = [0.] * self.__tree.num_nonleaf_nodes
        for i in range(self.__tree.num_nonleaf_nodes):
            ii[i] = s[i] - self.__padded_b[i].T @ y[i]

        # -> iii
        iii = self.__nonleaf_constraint.op_nonleaf(x, self.__tree.num_nonleaf_nodes, u)

        # -> iv
        iv = [np.zeros((num_si + 2, 1))] * self.__tree.num_nodes
        for i in range(1, self.__tree.num_nodes):
            anc = self.__tree.ancestor_of_node(i)
            half_t = t[i] * 0.5
            iv[i] = np.vstack((self.__sqrt_nonleaf_state_costs[i] @ x[anc],
                               self.__sqrt_nonleaf_input_costs[i] @ u[anc],
                               half_t, half_t))

        # -> v
        v = self.__leaf_constraint.op_leaf(x[self.__tree.num_nonleaf_nodes:], self.__tree.num_leaf_nodes)

        # -> vi
        vi = [np.zeros((self.__num_states + 2, 1))] * self.__tree.num_leaf_nodes
        for i in range(self.__tree.num_leaf_nodes):
            half_s = s[i + self.__tree.num_nonleaf_nodes] * 0.5
            vi[i] = np.vstack((self.__sqrt_leaf_state_costs[i] @ x[i + self.__tree.num_nonleaf_nodes],
                               half_s, half_s))

        # Gather dual
        dual = []
        for i in [i_, ii, iii, iv, v, vi]:
            dual += self.__flatten(i)

        return dual

    def __adj(self, dual):
        # Split dual into i_, ii, iii, iv, v, vi
        num_si = self.__num_states + self.__num_inputs
        idx = 0
        i_ = [np.array(dual[idx + i * self.__y_size:idx + i * self.__y_size + self.__y_size]).reshape(
            self.__y_size, 1) for i in range(self.__tree.num_nonleaf_nodes)]
        idx += self.__y_size * self.__tree.num_nonleaf_nodes
        ii = [np.array(dual[idx + i * 1:idx + i * 1 + 1]).reshape(1, 1) for i in range(self.__tree.num_nonleaf_nodes)]
        idx += self.__tree.num_nonleaf_nodes
        iii, idx = self.__nonleaf_constraint.assign_dual(dual, idx, self.__tree.num_nonleaf_nodes)
        iv_size = num_si + 2
        iv = [np.array(dual[idx + i * iv_size:idx + i * iv_size + iv_size]).reshape(
            iv_size, 1) for i in range(self.__tree.num_nodes)]
        idx += iv_size * self.__tree.num_nodes
        v, idx = self.__leaf_constraint.assign_dual(dual, idx, self.__tree.num_leaf_nodes)
        vi_size = self.__num_states + 2
        vi = [np.array(dual[idx + i * vi_size:idx + i * vi_size + vi_size]).reshape(
            vi_size, 1) for i in range(self.__tree.num_leaf_nodes)]

        # -> s (nonleaf)
        s = [0.] * self.__tree.num_nodes
        s[:self.__tree.num_nonleaf_nodes] = ii

        # -> y
        y = [np.zeros((self.__y_size, 1))] * self.__tree.num_nonleaf_nodes
        for i in range(self.__tree.num_nonleaf_nodes):
            y[i] = i_[i] - self.__padded_b[i] * ii[i]

        # -> x (nonleaf) and u:Gamma
        x = [np.zeros((self.__num_states, 1))] * self.__tree.num_nodes
        u = [np.zeros((self.__num_inputs, 1))] * self.__tree.num_nonleaf_nodes
        x, u = self.__nonleaf_constraint.adj_nonleaf(iii, x, self.__tree.num_nonleaf_nodes, u)
        # -> x (nonleaf) and u
        for i in range(1, self.__tree.num_nodes):
            anc = self.__tree.ancestor_of_node(i)
            x[anc] += self.__sqrt_nonleaf_state_costs[i] @ iv[i][:self.__num_states]
            u[anc] += self.__sqrt_nonleaf_input_costs[i] @ iv[i][self.__num_states:num_si]

        # -> t
        t = [0.] * self.__tree.num_nodes
        for i in range(1, self.__tree.num_nodes):
            t[i] = 0.5 * (iv[i][num_si] + iv[i][num_si + 1])

        # -> x (leaf):Gamma
        x[self.__tree.num_nonleaf_nodes:] = self.__leaf_constraint.adj_leaf(
            v, x[self.__tree.num_nonleaf_nodes:], self.__tree.num_leaf_nodes)

        # -> x (leaf)
        for i in range(self.__tree.num_leaf_nodes):
            x[i + self.__tree.num_nonleaf_nodes] += self.__sqrt_leaf_state_costs[i] @ vi[i][:self.__num_states]

        # -> s (leaf)
        for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            idx = i - self.__tree.num_nonleaf_nodes
            s[i] = 0.5 * (vi[idx][self.__num_states] + vi[idx][self.__num_states + 1])

        # Gather primal
        prim = []
        for i in [u, x, y, t, s]:
            prim += self.__flatten(i)

        return prim

    def __test_op_and_adj(self):
        # Create random primal
        f = 10
        u = [f * np.random.randn(self.__num_inputs, 1) for _ in range(self.__tree.num_nonleaf_nodes)]
        x = [f * np.random.randn(self.__num_states, 1) for _ in range(self.__tree.num_nodes)]
        y = [None] * self.__tree.num_nonleaf_nodes
        for i in range(self.__tree.num_nonleaf_nodes):
            num_ch = len(self.__tree.children_of_node(i))
            size_y = 2 * num_ch + 1
            y[i] = np.vstack(((f * np.random.randn(size_y, 1)), np.zeros((self.__y_size - size_y, 1))))
        t = [0.] + (f * np.random.randn(self.__tree.num_nodes - 1)).tolist()
        s = (f * np.random.randn(self.__tree.num_nodes)).tolist()
        prim = []
        for i in [u, x, y, t, s]:
            prim += self.__flatten(i)
        self.__prim_before_op = deepcopy(prim)
        dual = self.__op(prim)
        self.__dual_after_op_before_adj = deepcopy(dual)
        prim = self.__adj(dual)
        self.__prim_after_adj = deepcopy(prim)

    def __get_step_size(self):
        prim_size = (
                (self.__num_inputs + self.__y_size) * self.__tree.num_nonleaf_nodes +  # u, y
                (self.__num_states + 2) * self.__tree.num_nodes  # x, t, s
        )
        p = np.zeros(prim_size)
        d = self.__op(p)
        dual_size = len(d)
        op = LinearOperator(dtype=self.__tree.dtype, shape=(dual_size, prim_size), matvec=self.__op)
        adj = LinearOperator(dtype=self.__tree.dtype, shape=(prim_size, dual_size), matvec=self.__adj)
        adj_op = adj * op
        eigen, _ = eigs(adj_op)
        max_eigen = np.real(max(eigen))
        nrm = np.sqrt(max_eigen)
        nrm_recip = .99 / nrm
        self.__step_size = nrm_recip

    def __test_dot(self):
        # Compute result of x'My
        p = deepcopy(np.asarray(self.__prim_before_op).reshape(1, -1))[0]
        d = deepcopy(np.asarray(self.__dual_after_op_before_adj).reshape(1, -1))[0]
        for i in range(len(p)):
            p[i] = -3. if p[i] else 0.
        for i in range(len(d)):
            d[i] = 2. if d[i] else 0.
        x = np.hstack((p, d))
        op_p = np.asarray(self.__op(p)).reshape(1, -1)[0]
        adj_d = np.asarray(self.__adj(d)).reshape(1, -1)[0]
        m_y_p = p - self.__step_size * adj_d
        m_y_d = d - self.__step_size * op_p
        m_y = np.hstack((m_y_p, m_y_d))
        res = x @ m_y.T
        self.__dot_vector = x.tolist()
        self.__dot_result = res

    def __print(self):
        print("Problem Data\n"
              "+ Step size: ", self.__step_size, "\n")
        return self


class ProblemFactory:
    """
    Risk-averse optimal control problem builder
    """

    def __init__(self, scenario_tree: treeFactory.Tree, num_states, num_inputs):
        """
        :param scenario_tree: instance of ScenarioTree
        """
        self.__tree = scenario_tree
        self.__num_states = num_states
        self.__num_inputs = num_inputs
        self.__list_of_dynamics = [None] * self.__tree.num_nodes
        self.__list_of_nonleaf_state_costs = [None] * self.__tree.num_nodes
        self.__list_of_nonleaf_input_costs = [None] * self.__tree.num_nodes
        self.__list_of_leaf_state_costs = [None] * self.__tree.num_leaf_nodes
        self.__nonleaf_constraint = [None] * self.__tree.num_nonleaf_nodes
        self.__leaf_constraint = [None] * self.__tree.num_leaf_nodes
        self.__list_of_risks = [None] * self.__tree.num_nonleaf_nodes
        self.__test = False
        self.__preload_constraints()

    def __check_markovian(self, section):
        if not self.__tree.is_markovian:
            raise ValueError("Markovian " + section + " provided but scenario tree is not Markovian")

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------
    def with_markovian_linear_dynamics(self, state_dynamics, input_dynamics):
        self.__check_markovian("linear dynamics")
        num_events = len(state_dynamics)
        # check equal number of dynamics given
        if num_events != len(input_dynamics):
            raise ValueError("Different number of Markovian linear state and input dynamics given")
        # check dynamics are compatible
        if state_dynamics[0].shape[0] != input_dynamics[0].shape[0]:
            raise ValueError("Markovian linear state and input dynamics are incompatible")
        self.__list_of_dynamics[0] = build.Linear(np.zeros(state_dynamics[0].shape),
                                                  np.zeros(input_dynamics[0].shape))
        for i in range(1, self.__tree.num_nodes):
            event = self.__tree.event_of_node(i)
            self.__list_of_dynamics[i] = build.Linear(deepcopy(state_dynamics[event]),
                                                      deepcopy(input_dynamics[event]))
        return self

    def with_markovian_affine_dynamics(self, state_dynamics, input_dynamics, affine_dynamics):
        self.__check_markovian("affine dynamics")
        num_events = len(state_dynamics)
        # check equal number of dynamics given
        if num_events != len(input_dynamics):
            raise ValueError("Different number of Markovian affine state and input dynamics given")
        if num_events != len(affine_dynamics):
            raise ValueError("Different number of Markovian affine state and input dynamics given")
        # check dynamics are compatible
        if state_dynamics[0].shape[0] != input_dynamics[0].shape[0]:
            raise ValueError("Markovian affine state and input dynamics are incompatible")
        if state_dynamics[0].shape[0] != affine_dynamics[0].shape[0]:
            raise ValueError("Markovian affine state and affine dynamics are incompatible")
        self.__list_of_dynamics[0] = build.Affine(np.zeros(state_dynamics[0].shape),
                                                  np.zeros(input_dynamics[0].shape),
                                                  np.zeros(affine_dynamics[0].shape))
        for i in range(1, self.__tree.num_nodes):
            event = self.__tree.event_of_node(i)
            self.__list_of_dynamics[i] = build.Affine(deepcopy(state_dynamics[event]),
                                                      deepcopy(input_dynamics[event]),
                                                      deepcopy(affine_dynamics[event]))
        return self

    # --------------------------------------------------------
    # Costs
    # --------------------------------------------------------
    def with_markovian_nonleaf_costs(self, state_costs, input_costs):
        self.__check_markovian("costs")
        self.__list_of_nonleaf_state_costs[0] = np.zeros((state_costs[0].shape[0], state_costs[0].shape[1]))
        self.__list_of_nonleaf_input_costs[0] = np.zeros((input_costs[0].shape[0], input_costs[0].shape[1]))
        for i in range(1, self.__tree.num_nodes):
            event = self.__tree.event_of_node(i)
            self.__list_of_nonleaf_state_costs[i] = deepcopy(state_costs[event])
            self.__list_of_nonleaf_input_costs[i] = deepcopy(input_costs[event])
        return self

    def with_nonleaf_cost(self, state_cost, input_cost):
        try:
            self.__list_of_nonleaf_state_costs[0] = np.zeros((state_cost[0].shape[0], state_cost[0].shape[1]))
            self.__list_of_nonleaf_input_costs[0] = np.zeros((input_cost[0].shape[0], input_cost[0].shape[1]))
        except:
            self.__list_of_nonleaf_state_costs[0] = np.zeros((state_cost[0].shape[0], state_cost[0].shape[0]))
            self.__list_of_nonleaf_input_costs[0] = np.zeros((input_cost[0].shape[0], input_cost[0].shape[0]))
        for i in range(1, self.__tree.num_nodes):
            self.__list_of_nonleaf_state_costs[i] = deepcopy(state_cost)
            self.__list_of_nonleaf_input_costs[i] = deepcopy(input_cost)
        return self

    def with_leaf_cost(self, state_cost):
        for i in range(self.__tree.num_leaf_nodes):
            self.__list_of_leaf_state_costs[i] = deepcopy(state_cost)
        return self

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------
    def __preload_constraints(self):
        # load "No constraint"
        self.__nonleaf_constraint = build.No()
        self.__leaf_constraint = build.No()

    def with_nonleaf_constraint(self, state_input_constraint):
        self.__nonleaf_constraint = deepcopy(state_input_constraint)
        return self

    def with_leaf_constraint(self, state_constraint):
        self.__leaf_constraint = deepcopy(state_constraint)
        return self

    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------
    def with_risk(self, risk):
        for i in range(self.__tree.num_nonleaf_nodes):
            risk_i = deepcopy(risk)
            self.__list_of_risks[i] = risk_i.make_risk(self.__tree.cond_prob_of_children_of_node(i))
        return self

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------
    def with_tests(self):
        self.__test = True
        return self

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------
    def generate_problem(self):
        """
        Generates problem data from the given build
        """
        problem = Problem(self.__tree,
                          self.__num_states,
                          self.__num_inputs,
                          self.__list_of_dynamics,
                          self.__list_of_nonleaf_state_costs,
                          self.__list_of_nonleaf_input_costs,
                          self.__list_of_leaf_state_costs,
                          self.__nonleaf_constraint,
                          self.__leaf_constraint,
                          self.__list_of_risks,
                          self.__test)
        return problem
