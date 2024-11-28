import jinja2 as j2
import os
import numpy as np
import scipy as sp
from scipy.linalg import sqrtm
from scipy.sparse.linalg import LinearOperator, eigs
import cvxpy as cvx
from copy import deepcopy
from . import treeFactory
from . import build


class Problem:
    """
    Risk-averse optimal control problem and offline cache storage
    """

    def __init__(self, scenario_tree: treeFactory.Tree, num_states, num_inputs, state_dyn, input_dyn, state_cost,
                 input_cost, terminal_cost, nonleaf_constraint, leaf_constraint, risk, test):
        """
        :param scenario_tree: instance of ScenarioTree
        :param num_states: number of system states
        :param num_inputs: number of system inputs
        :param state_dyn: list of state dynamics matrices (size: num_nodes, used: 1 to num_nodes)
        :param input_dyn: list of input dynamics matrices (size: num_nodes, used: 1 to num_nodes)
        :param state_cost: list of state cost matrices (size: num_nodes, used: 1 to num_nodes)
        :param input_cost: list of input cost matrices (size: num_nodes, used: 1 to num_nodes)
        :param terminal_cost: list of terminal cost matrices (size: num_nodes, used: num_nonleaf_nodes to num_nodes)
        :param nonleaf_constraint: list of state-input constraint classes (size: num_nonleaf_nodes)
        :param leaf_constraint: list of input constraint classes (size: num_nodes, used: num_nonleaf_nodes to num_nodes)
        :param risk: list of risk classes (size: num_nonleaf_nodes, used: all)
        :param test: whether to compute test data

        Note: avoid using this constructor directly; use a factory instead
        """
        self.__tree = scenario_tree
        self.__num_states = num_states
        self.__num_inputs = num_inputs
        self.__list_of_state_dynamics = state_dyn
        self.__list_of_input_dynamics = input_dyn
        self.__list_of_state_input_dynamics = [np.hstack((A, B)) for A, B in zip(state_dyn, input_dyn)]
        self.__list_of_nonleaf_state_costs = state_cost
        self.__list_of_nonleaf_input_costs = input_cost
        self.__list_of_leaf_state_costs = terminal_cost
        self.__list_of_nonleaf_constraints = nonleaf_constraint
        self.__list_of_leaf_constraints = leaf_constraint
        self.__list_of_risks = risk
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

    # GETTERS
    def state_dynamics_at_node(self, idx):
        return self.__list_of_state_dynamics[idx]

    def input_dynamics_at_node(self, idx):
        return self.__list_of_input_dynamics[idx]

    def nonleaf_state_cost_at_node(self, idx):
        return self.__list_of_nonleaf_state_costs[idx]

    def nonleaf_input_cost_at_node(self, idx):
        return self.__list_of_nonleaf_input_costs[idx]

    def leaf_state_cost_at_node(self, idx):
        return self.__list_of_leaf_state_costs[idx]

    def nonleaf_constraint_at_node(self, idx):
        return self.__list_of_nonleaf_constraints[idx]

    def leaf_constraint_at_node(self, idx):
        return self.__list_of_leaf_constraints[idx]

    def risk_at_node(self, idx):
        return self.__list_of_risks[idx]

    def get_file(self, name):
        path = os.path.join(os.getcwd(), self.__tree.folder)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, name)

    @staticmethod
    def save_tensor(file, tensor):
        np.savetxt(file,
                   X=tensor.reshape(-1),
                   fmt='%-.15f',
                   delimiter='\n',
                   newline='\n',
                   header=f"{tensor.shape[1]}\n"
                          f"{tensor.shape[2]}\n"
                          f"{tensor.shape[0]}",
                   comments='')

    @staticmethod
    def save_vector(file, vector):
        np.savetxt(file,
                   X=vector,
                   fmt='%-.15f',
                   delimiter='\n',
                   newline='\n',
                   header=f"{len(vector)}\n"
                          f"{1}\n"
                          f"{1}",
                   comments='')

    def generate_problem_files(self):
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
                                 nonleaf_constraint=self.__list_of_nonleaf_constraints[0],
                                 leaf_constraint=self.__list_of_leaf_constraints[-1],
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
        stack_state_dyn = np.array(self.__list_of_state_dynamics)
        stack_input_dyn = np.array(self.__list_of_input_dynamics)
        stack_AB_dyn = np.array(self.__list_of_state_input_dynamics)
        stack_state_cost = np.array(self.__list_of_nonleaf_state_costs)
        stack_input_cost = np.array(self.__list_of_nonleaf_input_costs)
        stack_terminal_cost = np.array(self.__list_of_leaf_state_costs)
        stack_sqrt_state_cost = np.array(self.__sqrt_nonleaf_state_costs)
        stack_sqrt_input_cost = np.array(self.__sqrt_nonleaf_input_costs)
        stack_sqrt_terminal_cost = np.array(self.__sqrt_leaf_state_costs)
        stack_P = np.array(self.__P)
        stack_K = np.array(self.__K)
        stack_low_chol = np.array(self.__cholesky_lower)
        stack_dyn_tr = np.array(self.__sum_of_dynamics_tr)
        stack_APB = np.array(self.__At_P_B)
        stack_null = np.array(self.__nullspace_projection_matrix)
        stack_b = np.array(self.__padded_b)
        # Create tensor dict
        tensors = {
            "stateDyn": stack_state_dyn,
            "inputDyn": stack_input_dyn,
            "AB_dyn": stack_AB_dyn,
            "stateCost": stack_state_cost,
            "inputCost": stack_input_cost,
            "terminalCost": stack_terminal_cost,
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
        if self.__list_of_nonleaf_constraints[0].is_rectangle:
            stack_lb = np.array([con.lower_bound for con in self.__list_of_nonleaf_constraints])
            stack_ub = np.array([con.upper_bound for con in self.__list_of_nonleaf_constraints])
            tensors.update({
                "nonleafConstraintLB": stack_lb,
                "nonleafConstraintUB": stack_ub
            })
        if self.__list_of_leaf_constraints[0].is_rectangle:
            stack_lb = np.array([con.lower_bound for con in self.__list_of_leaf_constraints])
            stack_ub = np.array([con.upper_bound for con in self.__list_of_leaf_constraints])
            tensors.update({
                "leafConstraintLB": stack_lb,
                "leafConstraintUB": stack_ub
            })
        # Generate tensor files
        for name, tensor in tensors.items():
            self.save_tensor(self.get_file(name), tensor)
        if self.__test:
            stack_ker_con = np.array(self.__kernel_constraint_matrix)
            test_tensors = {
                "S2": stack_ker_con
            }
            test_vectors = {
                "dpTestStates": self.__dp_test_states,
                "dpTestInputs": self.__dp_test_inputs,
                "dpProjectedStates": self.__dp_projected_states,
                "dpProjectedInputs": self.__dp_projected_inputs,
                "primBeforeOp": self.__prim_before_op,
                "dualAfterOpBeforeAdj": self.__dual_after_op_before_adj,
                "primAfterAdj": self.__prim_after_adj,
                "dotVector": self.__dot_vector,
                "dotResult": [self.__dot_result]
            }
            print(self.__prim_before_op)
            for name, tensor in test_tensors.items():
                self.save_tensor(self.get_file(name), tensor)
            for name, vector in test_vectors.items():
                self.save_vector(self.get_file(name), vector)

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    def generate_offline(self):
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
                            + self.__list_of_input_dynamics[j].T @ self.__P[j] @ self.__list_of_input_dynamics[j]
                sum_for_k = sum_for_k \
                            + self.__list_of_input_dynamics[j].T @ self.__P[j] @ self.__list_of_state_dynamics[j]
            r_tilde = np.eye(self.__num_inputs) + sum_for_r
            self.__cholesky_lower[i] = sp.linalg.cholesky(r_tilde, lower=self.__lower, check_finite=True)
            self.__K[i] = sp.linalg.cho_solve((self.__cholesky_lower[i], self.__lower), -sum_for_k)
            sum_for_p = 0
            for j in children_of_i:
                sum_of_dynamics = self.__list_of_state_dynamics[j] + self.__list_of_input_dynamics[j] @ self.__K[i]
                self.__sum_of_dynamics_tr[j] = sum_of_dynamics.T
                sum_for_p = sum_for_p + self.__sum_of_dynamics_tr[j] @ self.__P[j] @ sum_of_dynamics
                self.__At_P_B[j] = self.__sum_of_dynamics_tr[j] @ self.__P[j] @ self.__list_of_input_dynamics[j]
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
                                self.__list_of_state_dynamics[ch] @ x[:, node] +
                                self.__list_of_input_dynamics[ch] @ u[:, node]]

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
        ii = [None] * self.__tree.num_nonleaf_nodes
        for i in range(self.__tree.num_nonleaf_nodes):
            ii[i] = s[i] - np.asarray(self.__padded_b[i]).T @ y[i]

        # -> iii
        iii = None
        if self.__list_of_nonleaf_constraints[0].is_no:
            iii = np.array([]).reshape(1, 0)
        elif self.__list_of_nonleaf_constraints[0].is_rectangle:
            # Gamma_{x} and Gamma{u} do not change x and u
            iii = [None] * self.__tree.num_nonleaf_nodes
            for i in range(self.__tree.num_nonleaf_nodes):
                iii[i] = np.vstack((x[i], u[i]))
        elif self.__list_of_nonleaf_constraints[0].is_ball:
            pass  # TODO!
        else:
            raise ValueError("Constraint not supported.")

        # -> iv
        iv = [np.zeros((num_si + 2, 1))] * self.__tree.num_nodes
        for i in range(1, self.__tree.num_nodes):
            anc = self.__tree.ancestor_of_node(i)
            half_t = t[i] * 0.5
            iv[i] = np.vstack((self.__sqrt_nonleaf_state_costs[i] @ x[anc],
                               self.__sqrt_nonleaf_input_costs[i] @ u[anc],
                               half_t, half_t))

        # -> v
        v = None
        if self.__list_of_leaf_constraints[0].is_no:
            v = np.array([]).reshape(1, 0)
        elif self.__list_of_leaf_constraints[0].is_rectangle:
            # Gamma_{x} does not change x
            v = [np.zeros((self.__num_states, 1))] * self.__tree.num_leaf_nodes
            for i in range(self.__tree.num_leaf_nodes):
                v[i] = x[i + self.__tree.num_nonleaf_nodes]
        elif self.__list_of_leaf_constraints[0].is_ball:
            pass  # TODO!
        else:
            raise ValueError("Constraint not supported.")

        # -> vi
        vi = [np.zeros((self.__num_states + 2, 1))] * self.__tree.num_leaf_nodes
        for i in range(self.__tree.num_leaf_nodes):
            half_s = s[i] * 0.5
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
        if self.__list_of_nonleaf_constraints[0].is_no:
            iii = np.array([]).reshape(1, 0)
        elif self.__list_of_nonleaf_constraints[0].is_rectangle:
            iii = [np.array(dual[idx + i * num_si:idx + i * num_si + num_si]).reshape(
                num_si, 1) for i in range(self.__tree.num_nonleaf_nodes)]
            idx += num_si * self.__tree.num_nonleaf_nodes
        elif self.__list_of_nonleaf_constraints[0].is_ball:
            pass  # TODO!
        else:
            raise ValueError("Constraint not supported.")
        iv_size = num_si + 2
        iv = [np.array(dual[idx + i * iv_size:idx + i * iv_size + iv_size]).reshape(
            iv_size, 1) for i in range(self.__tree.num_nodes)]
        idx += iv_size * self.__tree.num_nodes
        if self.__list_of_leaf_constraints[0].is_no:
            v = np.array([]).reshape(1, 0)
        elif self.__list_of_leaf_constraints[0].is_rectangle:
            v = [np.array(dual[idx + i * self.__num_states:idx + i * self.__num_states + self.__num_states]).reshape(
                self.__num_states, 1) for i in range(self.__tree.num_leaf_nodes)]
            idx += self.__num_states * self.__tree.num_leaf_nodes
        elif self.__list_of_leaf_constraints[0].is_ball:
            pass  # TODO!
        else:
            raise ValueError("Constraint not supported.")
        vi_size = self.__num_states + 2
        vi = [np.array(dual[idx + i * vi_size:idx + i * vi_size + vi_size]).reshape(
            vi_size, 1) for i in range(self.__tree.num_leaf_nodes)]

        # -> s (nonleaf)
        s = [None] * self.__tree.num_nodes
        s[:self.__tree.num_nonleaf_nodes] = ii[:self.__tree.num_nonleaf_nodes]

        # -> y
        y = [np.zeros((self.__y_size, 1))] * self.__tree.num_nonleaf_nodes
        for i in range(self.__tree.num_nonleaf_nodes):
            y[i] = i_[i] - np.asarray(self.__padded_b[i]) * ii[i]

        # -> x (nonleaf) and u:Gamma
        x = [np.zeros((self.__num_states, 1))] * self.__tree.num_nodes
        u = [np.zeros((self.__num_inputs, 1))] * self.__tree.num_nonleaf_nodes
        if self.__list_of_nonleaf_constraints[0].is_no:
            pass
        elif self.__list_of_nonleaf_constraints[0].is_rectangle:
            # Gamma_{x} and Gamma{u} do not change x and u
            for i in range(self.__tree.num_nonleaf_nodes):
                x[i] = iii[i][:self.__num_states]
                u[i] = iii[i][self.__num_states:num_si]
        elif self.__list_of_nonleaf_constraints[0].is_ball:
            pass  # TODO!
        else:
            raise ValueError("Constraint not supported.")

        # -> x (nonleaf) and u
        for i in range(1, self.__tree.num_nodes):
            anc = self.__tree.ancestor_of_node(i)
            x[anc] += self.__sqrt_nonleaf_state_costs[i] @ iv[i][:self.__num_states]
            u[anc] += self.__sqrt_nonleaf_input_costs[i] @ iv[i][self.__num_states:num_si]

        # -> t
        t = [0.] + [None] * (self.__tree.num_nodes - 1)
        for i in range(1, self.__tree.num_nodes):
            t[i] = 0.5 * (iv[i][num_si] + iv[i][num_si + 1])

        # -> x (leaf):Gamma
        if self.__list_of_leaf_constraints[0].is_no:
            pass
        elif self.__list_of_leaf_constraints[0].is_rectangle:
            for i in range(self.__tree.num_leaf_nodes):
                x[i + self.__tree.num_nonleaf_nodes] = v[i]
        elif self.__list_of_leaf_constraints[0].is_ball:
            pass  # TODO!
        else:
            raise ValueError("Constraint not supported.")

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
        op = LinearOperator(dtype=None, shape=(dual_size, prim_size), matvec=self.__op)
        adj = LinearOperator(dtype=None, shape=(prim_size, dual_size), matvec=self.__adj)
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

    def print(self):
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
        self.__list_of_state_dynamics = [None] * self.__tree.num_nodes
        self.__list_of_input_dynamics = [None] * self.__tree.num_nodes
        self.__list_of_nonleaf_state_costs = [None] * self.__tree.num_nodes
        self.__list_of_nonleaf_input_costs = [None] * self.__tree.num_nodes
        self.__list_of_leaf_state_costs = [None] * self.__tree.num_leaf_nodes
        self.__list_of_nonleaf_constraints = [None] * self.__tree.num_nonleaf_nodes
        self.__list_of_leaf_constraints = [None] * self.__tree.num_leaf_nodes
        self.__list_of_risks = [None] * self.__tree.num_nonleaf_nodes
        self.__test = False
        self.__preload_constraints()

    def __check_markovian(self, section):
        if not self.__tree.is_markovian:
            raise ValueError("Markovian " + section + " provided but scenario tree is not Markovian")

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------
    def with_markovian_dynamics(self, state_dynamics, input_dynamics):
        self.__check_markovian("dynamics")
        num_events = len(state_dynamics)
        # check equal number of dynamics given
        if num_events != len(input_dynamics):
            raise ValueError("Different number of Markovian state and input dynamics given")
        for i in range(1, num_events):
            # check all state dynamics have same shape
            if state_dynamics[i].shape != state_dynamics[0].shape:
                raise ValueError("Markovian state dynamics matrices are different shapes")
            # check all control dynamics have same shape
            if input_dynamics[i].shape != input_dynamics[0].shape:
                raise ValueError("Markovian input dynamics matrices are different shapes")
        self.__list_of_state_dynamics[0] = np.zeros((state_dynamics[0].shape[0], state_dynamics[0].shape[1]))
        self.__list_of_input_dynamics[0] = np.zeros((input_dynamics[0].shape[0], input_dynamics[0].shape[1]))
        for i in range(1, self.__tree.num_nodes):
            event = self.__tree.event_of_node(i)
            self.__list_of_state_dynamics[i] = deepcopy(state_dynamics[event])
            self.__list_of_input_dynamics[i] = deepcopy(input_dynamics[event])
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
        # load "No constraint" into constraints list
        for i in range(self.__tree.num_nodes):
            if i < self.__tree.num_nonleaf_nodes:
                self.__list_of_nonleaf_constraints[i] = build.No()
            if i < self.__tree.num_leaf_nodes:
                self.__list_of_leaf_constraints[i] = build.No()

    def with_nonleaf_constraint(self, state_input_constraint):
        for i in range(self.__tree.num_nonleaf_nodes):
            self.__list_of_nonleaf_constraints[i] = deepcopy(state_input_constraint)
        return self

    def with_leaf_constraint(self, state_constraint):
        for i in range(self.__tree.num_leaf_nodes):
            self.__list_of_leaf_constraints[i] = deepcopy(state_constraint)
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
                          self.__list_of_state_dynamics,
                          self.__list_of_input_dynamics,
                          self.__list_of_nonleaf_state_costs,
                          self.__list_of_nonleaf_input_costs,
                          self.__list_of_leaf_state_costs,
                          self.__list_of_nonleaf_constraints,
                          self.__list_of_leaf_constraints,
                          self.__list_of_risks,
                          self.__test)
        print("Computing offline data...")
        problem.generate_offline()
        print("Generating problem files...")
        problem.generate_problem_files()
        problem.print()
        return problem
