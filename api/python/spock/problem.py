import jinja2 as j2
import os
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator, eigs
from copy import deepcopy
from . import tree
from . import build

try:
    import cvxpy
except ImportError:
    cvxpy = None  # cvxpy is only needed for testing purposes


class Problem:
    """
    Risk-averse optimal control problem and offline cache storage
    """

    def __init__(self, scenario_tree: tree.Tree, num_states, num_inputs, dynamics,
                 nonleaf_cost, leaf_cost, nonleaf_constraint, leaf_constraint, risk,
                 precondition=False, test=False, julia=False):
        """
        :param scenario_tree: instance of ScenarioTree
        :param num_states: number of system states
        :param num_inputs: number of system inputs
        :param dynamics: list of dynamics (size: num_events)
        :param nonleaf_cost: list of nonleaf costs (size: num_nodes)
        :param leaf_cost: list of leaf costs (size: num_leaf_nodes)
        :param nonleaf_constraint: state-input constraint class (size: 1)
        :param leaf_constraint: state constraint class (size: 1)
        :param risk: list of risk classes (size: num_events)
        :param precondition: whether to precondition problem data
        :param test: whether to compute test data
        :param julia: whether to compute julia data

        Note: avoid using this constructor directly; use a factory instead
        """
        self.__tree = scenario_tree
        self.__num_states = num_states
        self.__num_inputs = num_inputs
        self.__list_of_dynamics = dynamics
        self.__nonleaf_cost = nonleaf_cost
        self.__leaf_cost = leaf_cost
        self.__list_of_risks = risk
        self.__nonleaf_constraint = nonleaf_constraint
        self.__leaf_constraint = leaf_constraint
        self.__precondition_requested = precondition
        self.__precondition_worth_it = False
        self.__test = test
        self.__julia = julia
        # Dynamics projection
        self.__P = [np.zeros((self.__num_states, self.__num_states)) for _ in range(self.__tree.num_nodes)]
        self.__q = [np.zeros((self.__num_states, 1)) for _ in range(self.__tree.num_nodes)]
        self.__K = [np.zeros((self.__num_states, self.__num_states)) for _ in range(self.__tree.num_nonleaf_nodes)]
        self.__d = [np.zeros((self.__num_states, 1)) for _ in range(self.__tree.num_nonleaf_nodes)]
        self.__lower = True  # Do not change!
        self.__cholesky_lower = [None for _ in range(self.__tree.num_nonleaf_nodes)]
        self.__sum_of_dynamics_tr = [np.zeros((num_states, num_states)) for _ in range(self.__tree.num_nodes)]  # A+B@K
        self.__At_P_B = [np.zeros((num_states, num_inputs)) for _ in range(self.__tree.num_nodes)]  # At@P@B
        self.__dp_test_init_state = []
        self.__dp_test_states = []
        self.__dp_test_inputs = []
        self.__dp_projected_states = []
        self.__dp_projected_inputs = []
        # Kernel projection
        self.__kernel_constraint_matrix = [np.zeros((0, 0)) for _ in range(self.__tree.num_nonleaf_nodes)]
        self.__nullspace_projection_matrix = [np.zeros((0, 0)) for _ in range(self.__tree.num_nonleaf_nodes)]
        self.__kernel_constraint_matrix_rows = []
        self.__max_nullspace_dim = self.__tree.num_events * 4 + 1
        self.__projected_ns = [np.zeros((0, 0)) for _ in range(self.__tree.num_nonleaf_nodes)]
        # L operator and adjoint testing
        self.__prim_before_op = []
        self.__dual_after_op_before_adj = []
        self.__prim_after_adj = []
        self.__prim_random = None
        self.__dual_random = None
        self.__test_is_adjoint_result = None
        self.__dot_vector = []
        self.__dot_result = []
        # Other
        self.__y_size = 2 * self.__tree.num_events + 1
        self.__step_size = 0
        self.__prim_size = 0
        self.__dual_size = 0
        self.__padded_b = [np.zeros((0, 0)) for _ in range(self.__tree.num_nonleaf_nodes)]
        self.__scaling = None
        # Generate data and files
        print("Computing offline data...")
        self.__generate_offline()
        print("Generating problem files...")
        self.__generate_problem_files()

    # GETTERS
    @property
    def num_states(self):
        return self.__num_states

    @property
    def num_inputs(self):
        return self.__num_inputs

    def dynamics_at_node(self, idx):
        return self.__list_of_dynamics[idx]

    def nonleaf_cost(self):
        return self.__nonleaf_cost

    def leaf_cost(self):
        return self.__leaf_cost

    def nonleaf_constraint(self):
        return self.__nonleaf_constraint

    def leaf_constraint(self):
        return self.__leaf_constraint

    def risk_at_node(self, idx):
        return self.__list_of_risks[idx]

    @property
    def size_prim(self):
        return self.__prim_size

    @property
    def size_dual(self):
        return self.__dual_size

    @property
    def scaling(self):
        return self.__scaling

    @property
    def preconditioned(self):
        return self.__precondition_worth_it

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    def __generate_offline(self):
        if self.__precondition_requested:  # Must be first
            scale_x, scale_u = self.__preconditioning_check()
            if self.__precondition_worth_it:
                self.__preconditioning(scale_x, scale_u)
            else:
                print("Not worth preconditioning! Continuing without...")
        self.__offline_projection_dynamics()
        self.__offline_projection_kernel()
        self.__pad_b()
        self.__get_step_size()
        if self.__test:  # Must be after `get_step_size()`
            self.__test_dynamic_programming()
            self.__test_op_and_adj()
            self.__test_is_adjoint()
            self.__test_dot()

    def __preconditioning_check(self):
        """
        Find scaling parameters to improve step size.
        If all scaling parameters are <=1, preconditioning is not worth it.
        Caution! Only use diagonal scaling matrices.
        """
        scale_x = np.ones(self.__num_states)
        scale_u = np.ones(self.__num_inputs)
        scale_x = self.__nonleaf_cost.get_scaling_states(scale_x)
        scale_u = self.__nonleaf_cost.get_scaling_inputs(scale_u)
        scale_x = self.__leaf_cost.get_scaling_states(scale_x)
        mul = np.sqrt(self.__tree.max_num_children)
        scale_x *= mul
        scale_u *= mul
        self.__precondition_worth_it = not (np.allclose(scale_x, 1.) and np.allclose(scale_u, 1.))
        return scale_x, scale_u

    def __preconditioning(self, scale_x, scale_u):
        """
        Condition problem data with scaling parameters.
        """
        self.__scaling = np.hstack((scale_x, scale_u))
        if not (self.__scaling.all() > 0.):
            raise Exception("Preconditioning parameters are invalid!")
        scale_x_inv = np.diag(1 / scale_x)
        scale_u_inv = np.diag(1 / scale_u)
        scale_x_mat = np.diag(scale_x)
        self.__list_of_dynamics = [d.condition(scale_x_inv, scale_u_inv, scale_x_mat) for d in self.__list_of_dynamics]
        self.__nonleaf_cost.condition(scale_x_inv, scale_u_inv)
        self.__leaf_cost.condition(scale_x_inv)
        self.__nonleaf_constraint.condition(np.diag(self.__scaling))
        self.__leaf_constraint.condition(scale_x_mat)

    def __offline_projection_dynamics(self):
        for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            self.__P[i] = np.eye(self.__num_states)
        for i in reversed(range(self.__tree.num_nonleaf_nodes)):
            children_of_i = self.__tree.children_of_node(i)
            sum_for_r = 0
            sum_for_k = 0
            for j in children_of_i:
                sum_for_r = sum_for_r \
                            + self.__list_of_dynamics[j].B.T @ self.__P[j] @ self.__list_of_dynamics[j].B
                sum_for_k = sum_for_k \
                            + self.__list_of_dynamics[j].B.T @ self.__P[j] @ self.__list_of_dynamics[j].A
            r_tilde = np.eye(self.__num_inputs) + sum_for_r
            self.__cholesky_lower[i] = sp.linalg.cholesky(r_tilde, lower=self.__lower, check_finite=True)
            self.__K[i] = sp.linalg.cho_solve((self.__cholesky_lower[i], self.__lower), -sum_for_k)
            sum_for_p = 0
            for j in children_of_i:
                sum_of_dynamics = self.__list_of_dynamics[j].A + self.__list_of_dynamics[j].B @ self.__K[i]
                self.__sum_of_dynamics_tr[j] = sum_of_dynamics.T
                sum_for_p = sum_for_p + self.__sum_of_dynamics_tr[j] @ self.__P[j] @ sum_of_dynamics
                self.__At_P_B[j] = self.__sum_of_dynamics_tr[j] @ self.__P[j] @ self.__list_of_dynamics[j].B
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
        x = cvxpy.Variable((self.__num_states, self.__tree.num_nodes))
        u = cvxpy.Variable((self.__num_inputs, self.__tree.num_nonleaf_nodes))
        self.__dp_test_init_state = x_bar[:, 0]
        # Sum problem objectives and concatenate constraints
        cost = 0
        constraints = [x[:, 0] == x_bar[:, 0]]
        # Nonleaf nodes
        for node in range(self.__tree.num_nonleaf_nodes):
            cost += cvxpy.sum_squares(x[:, node] - x_bar[:, node]) + cvxpy.sum_squares(u[:, node] - u_bar[:, node])
            for ch in self.__tree.children_of_node(node):
                constraints += [x[:, ch] ==
                                self.__list_of_dynamics[ch].A @ x[:, node] +
                                self.__list_of_dynamics[ch].B @ u[:, node] +
                                self.__list_of_dynamics[ch].c.reshape(-1)]  # affine=zeros if linear dynamics

        # Leaf nodes
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            cost += cvxpy.sum_squares(x[:, node] - x_bar[:, node])

        problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
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

        # -> i
        i_ = deepcopy(y)

        # -> ii
        ii = [0. for _ in range(self.__tree.num_nonleaf_nodes)]
        for i in range(self.__tree.num_nonleaf_nodes):
            ii[i] = s[i] - self.__padded_b[i].T @ y[i]

        # -> iii
        iii = self.__nonleaf_constraint.op_nonleaf(x, self.__tree.num_nonleaf_nodes, u)

        # -> iv
        iv = self.__nonleaf_cost.op_nonleaf(x, u, t, self.__tree.ancestors())

        # -> v
        v = self.__leaf_constraint.op_leaf(x[self.__tree.num_nonleaf_nodes:], self.__tree.num_leaf_nodes)

        # -> vi
        vi = self.__leaf_cost.op_leaf(x[self.__tree.num_nonleaf_nodes:], s[self.__tree.num_nonleaf_nodes:])

        # Gather dual
        dual = []
        for i in [i_, ii, iii, iv, v, vi]:
            dual += self.__flatten(i)

        return dual

    def __adj(self, dual):
        # Split dual into i_, ii, iii, iv, v, vi
        idx = 0
        i_ = [np.array(dual[idx + i * self.__y_size:idx + i * self.__y_size + self.__y_size]).reshape(
            self.__y_size, 1) for i in range(self.__tree.num_nonleaf_nodes)]
        idx += self.__y_size * self.__tree.num_nonleaf_nodes
        ii = [np.array(dual[idx + i * 1:idx + i * 1 + 1]).reshape(1, 1) for i in range(self.__tree.num_nonleaf_nodes)]
        idx += self.__tree.num_nonleaf_nodes
        iii, idx = self.__nonleaf_constraint.assign_dual(dual, idx, self.__tree.num_nonleaf_nodes)
        iv, idx = self.__nonleaf_cost.assign_dual(dual, idx)
        v, idx = self.__leaf_constraint.assign_dual(dual, idx, self.__tree.num_leaf_nodes)
        vi, idx = self.__leaf_cost.assign_dual(dual, idx)

        # -> s (nonleaf)
        s = [0. for _ in range(self.__tree.num_nodes)]
        s[:self.__tree.num_nonleaf_nodes] = ii

        # -> y
        y = [np.zeros((self.__y_size, 1)) for _ in range(self.__tree.num_nonleaf_nodes)]
        for i in range(self.__tree.num_nonleaf_nodes):
            y[i] = i_[i] - self.__padded_b[i] * ii[i]

        # -> x (nonleaf) and u:Gamma
        x = [np.zeros((self.__num_states, 1)) for _ in range(self.__tree.num_nodes)]
        u = [np.zeros((self.__num_inputs, 1)) for _ in range(self.__tree.num_nonleaf_nodes)]
        x, u = self.__nonleaf_constraint.adj_nonleaf(iii, x, self.__tree.num_nonleaf_nodes, u)
        # -> x (nonleaf), u and t
        x_, u_, t = self.__nonleaf_cost.adj_nonleaf(iv, self.__num_states, self.__num_inputs, self.__tree.ancestors())
        for i in range(self.__tree.num_nonleaf_nodes):
            x[i] += x_[i]
            u[i] += u_[i]

        # -> x (leaf):Gamma
        x[self.__tree.num_nonleaf_nodes:] = self.__leaf_constraint.adj_leaf(
            v, x[self.__tree.num_nonleaf_nodes:], self.__tree.num_leaf_nodes)

        # -> x (leaf) and s (leaf)
        x_, s_ = self.__leaf_cost.adj_leaf(vi, self.__num_states)
        for i in range(self.__tree.num_leaf_nodes):
            idx = i + self.__tree.num_nonleaf_nodes
            x[idx] += x_[i]
            s[idx] += s_[i]

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
        y = [None for _ in range(self.__tree.num_nonleaf_nodes)]
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

    def __test_is_adjoint(self):
        m = 100
        self.__prim_random = m * np.random.randn(self.__prim_size, 1)
        self.__dual_random = m * np.random.randn(self.__dual_size, 1)
        # y'Lx
        Lx = np.array(self.__op(self.__prim_random)).reshape(-1, 1)
        uno = (self.__dual_random.T @ Lx)[0, 0]
        # (L'y)'x
        Ly = np.array(self.__adj(self.__dual_random)).reshape(-1, 1)
        dos = (Ly.T @ self.__prim_random)[0, 0]
        if np.linalg.norm(uno - dos) > 1e-4:
            raise Exception("[test_is_adjoint] Adjoint does not match operator.\n")
        self.__test_is_adjoint_result = uno

    def __get_step_size(self):
        self.__prim_size = (
                (self.__num_inputs + self.__y_size) * self.__tree.num_nonleaf_nodes +  # u, y
                (self.__num_states + 2) * self.__tree.num_nodes  # x, t, s
        )
        p = np.zeros(self.__prim_size)
        d = self.__op(p)
        self.__dual_size = len(d)
        op = LinearOperator(dtype=self.__tree.dtype, shape=(self.__dual_size, self.__prim_size), matvec=self.__op)
        adj = LinearOperator(dtype=self.__tree.dtype, shape=(self.__prim_size, self.__dual_size), matvec=self.__adj)
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

    def __generate_problem_files(self):
        # Setup jinja environment
        template_dir = os.path.dirname(os.path.abspath(__file__))
        file_loader = j2.FileSystemLoader(template_dir)
        env = j2.Environment(loader=file_loader,
                             trim_blocks=True,
                             lstrip_blocks=True,
                             block_start_string="\% ",
                             block_end_string="%\\",
                             variable_start_string="\~",
                             variable_end_string="~\\",
                             comment_start_string="\#",
                             comment_end_string="#\\")
        template = env.get_template("jinja2.py")
        output = template.render(num_events=self.__tree.num_events,
                                 num_nonleaf_nodes=self.__tree.num_nonleaf_nodes,
                                 num_nodes=self.__tree.num_nodes,
                                 num_stages=self.__tree.num_stages,
                                 num_states=self.__num_states,
                                 num_inputs=self.__num_inputs,
                                 dynamics=self.__list_of_dynamics[0],
                                 nonleaf_cost=self.__nonleaf_cost,
                                 leaf_cost=self.__leaf_cost,
                                 nonleaf_constraint=self.__nonleaf_constraint,
                                 leaf_constraint=self.__leaf_constraint,
                                 risk=self.__list_of_risks[0],
                                 ker_con_rows=self.__kernel_constraint_matrix_rows,
                                 null_dim=self.__max_nullspace_dim,
                                 step_size=self.__step_size,
                                 precondition=self.__precondition_worth_it,
                                 )
        path = self.__tree.get_folder_path
        output_file = os.path.join(path, "data.json")
        fh = open(output_file, "w")
        fh.write(output)
        fh.close()
        # Generate stacks
        stack_input_dyn_tr = np.dstack([a.B.T for a in self.__list_of_dynamics])
        stack_AB_dyn = np.dstack([a.A_B for a in self.__list_of_dynamics])
        stack_P = np.dstack(self.__P)
        stack_K = np.dstack(self.__K)
        stack_dyn_tr = np.dstack(self.__sum_of_dynamics_tr)
        stack_APB = np.dstack(self.__At_P_B)
        stack_low_chol = np.dstack(self.__cholesky_lower)
        stack_null = np.dstack(self.__nullspace_projection_matrix)
        stack_b = np.dstack(self.__padded_b)
        l_con = [self.__nonleaf_constraint, self.__leaf_constraint]
        l_txt = ["nonleafConstraint", "leafConstraint"]
        # Create tensor and vector dicts
        tensors = {}
        tensors.update({
            "dynamics_BTr": stack_input_dyn_tr,
            "dynamics_AB": stack_AB_dyn,
            "dynamics_P": stack_P,
            "dynamics_K": stack_K,
            "dynamics_ABKTr": stack_dyn_tr,
            "dynamics_APB": stack_APB,
            "dynamics_lowCholesky": stack_low_chol,
            "risk_NNtr": stack_null,
            "risk_b": stack_b
        })
        if self.__list_of_dynamics[0].is_affine:
            stack_dyn_const = np.dstack([dyn.c for dyn in self.__list_of_dynamics])
            tensors.update({
                "dynamics_c": stack_dyn_const,
            })
        if self.__precondition_worth_it:
            stack_scaling = self.__scaling.reshape(-1)
            tensors.update({
                "scaling": stack_scaling,
            })
        cost = self.__nonleaf_cost
        cost_txt_nl = "nonleafCost"
        if cost.is_linear:
            stack_gradient_cost = np.dstack(cost.cost_gradient)
            tensors.update({
                cost_txt_nl + "_gradient": stack_gradient_cost,
            })
        else:
            stack_sqrt_state_cost = np.dstack(cost.Q_sqrt)
            stack_sqrt_input_cost = np.dstack(cost.R_sqrt)
            stack_nonleaf_translation = np.dstack(cost.translation)
            tensors.update({
                cost_txt_nl + "_sqrtQ": stack_sqrt_state_cost,
                cost_txt_nl + "_sqrtR": stack_sqrt_input_cost,
                cost_txt_nl + "_translation": stack_nonleaf_translation,
            })
        cost = self.__leaf_cost
        cost_txt_l = "leafCost"
        if cost.is_linear:
            stack_gradient_terminal_cost = np.dstack([cost.cost_gradient[-1]])
            tensors.update({
                cost_txt_l + "_gradient": stack_gradient_terminal_cost,
            })
        else:
            stack_sqrt_terminal_cost = np.dstack([cost.Q_sqrt[-1]])
            stack_leaf_translation = np.dstack([cost.translation[-1]])
            tensors.update({
                cost_txt_l + "_sqrtQ": stack_sqrt_terminal_cost,
                cost_txt_l + "_translation": stack_leaf_translation,
            })
        for i in range(len(l_con)):
            con = l_con[i]
            txt = l_txt[i]
            if con.is_rectangle:
                tensors.update({
                    txt + "ILB": con.lower_bound,
                    txt + "IUB": con.upper_bound,
                })
            if con.is_polyhedron:
                tensors.update({
                    txt + "Gamma": con.matrix,
                    txt + "GLB": con.lower_bound,
                    txt + "GUB": con.upper_bound,
                })
            if con.is_polyhedron_with_identity:
                tensors.update({
                    txt + "ILB": con.rect.lower_bound,
                    txt + "IUB": con.rect.upper_bound,
                    txt + "Gamma": con.poly.matrix,
                    txt + "GLB": con.poly.lower_bound,
                    txt + "GUB": con.poly.upper_bound,
                })
        if self.__test:
            stack_ker_con = np.dstack(self.__kernel_constraint_matrix)
            tensors.update({
                "test_S2": stack_ker_con,
            })
            tensors.update({
                "test_dpOgStates": self.__dp_test_states.reshape(-1),
                "test_dpOgInputs": self.__dp_test_inputs.reshape(-1),
                "test_dpProjectedStates": self.__dp_projected_states.reshape(-1),
                "test_dpProjectedInputs": self.__dp_projected_inputs.reshape(-1),
                "test_primBeforeOp": np.array(self.__prim_before_op),
                "test_dualAfterOpBeforeAdj": np.array(self.__dual_after_op_before_adj),
                "test_primAfterAdj": np.array(self.__prim_after_adj),
                "test_adjRandomPrim": self.__prim_random,
                "test_adjRandomDual": self.__dual_random,
                "test_adjRandomResult": np.array([self.__test_is_adjoint_result]),
                "test_dotVector": np.array(self.__dot_vector),
                "test_dotResult": np.array([self.__dot_result]),
            })
        if self.__julia:
            prefix = "uncond_"
            stack_dyn_A = np.dstack([dyn.A_uncond for dyn in self.__list_of_dynamics])
            stack_dyn_B = np.dstack([dyn.B_uncond for dyn in self.__list_of_dynamics])
            stack_dyn_c = np.dstack([dyn.c_uncond for dyn in self.__list_of_dynamics])
            tensors.update({
                prefix + "dynamics_A": stack_dyn_A,
                prefix + "dynamics_B": stack_dyn_B,
                prefix + "dynamics_c": stack_dyn_c,
            })
            try:
                if self.__nonleaf_cost.Q_uncond[-1] is not None:
                    stack_cost_nonleaf_Q = np.dstack(self.__nonleaf_cost.Q_uncond)
                    tensors.update({
                        prefix + "cost_nonleaf_Q": stack_cost_nonleaf_Q,
                    })
            except:
                pass
            try:
                if self.__nonleaf_cost.R_uncond[-1] is not None:
                    stack_cost_nonleaf_R = np.dstack(self.__nonleaf_cost.R_uncond)
                    tensors.update({
                        prefix + "cost_nonleaf_R": stack_cost_nonleaf_R,
                    })
            except:
                pass
            try:
                if self.__nonleaf_cost.q_uncond[-1] is not None:
                    stack_cost_nonleaf_q = np.dstack(self.__nonleaf_cost.q_uncond)
                    tensors.update({
                        prefix + "cost_nonleaf_q": stack_cost_nonleaf_q,
                    })
            except:
                pass
            try:
                if self.__nonleaf_cost.r_uncond[-1] is not None:
                    stack_cost_nonleaf_r = np.dstack(self.__nonleaf_cost.r_uncond)
                    tensors.update({
                        prefix + "cost_nonleaf_r": stack_cost_nonleaf_r,
                    })
            except:
                pass
            try:
                if self.__leaf_cost.Q_uncond is not None:
                    stack_cost_leaf_Q = np.dstack([self.__leaf_cost.Q_uncond[-1]])
                    tensors.update({
                        prefix + "cost_leaf_Q": stack_cost_leaf_Q,
                    })
            except:
                pass
            try:
                if self.__leaf_cost.q_uncond is not None:
                    stack_cost_leaf_q = np.dstack([self.__leaf_cost.q_uncond[-1]])
                    tensors.update({
                        prefix + "cost_leaf_q": stack_cost_leaf_q,
                    })
            except:
                pass
            for i in range(len(l_con)):
                con = l_con[i]
                txt = l_txt[i]
                if con.is_rectangle:
                    tensors.update({
                        prefix + txt + "ILB": con.lower_bound_uncond,
                        prefix + txt + "IUB": con.upper_bound_uncond,
                    })
                if con.is_polyhedron:
                    tensors.update({
                        prefix + txt + "Gamma": con.matrix_uncond,
                        prefix + txt + "GLB": con.lower_bound_uncond,
                        prefix + txt + "GUB": con.upper_bound_uncond,
                    })
                if con.is_polyhedron_with_identity:
                    tensors.update({
                        prefix + txt + "ILB": con.rect.lower_bound_uncond,
                        prefix + txt + "IUB": con.rect.upper_bound_uncond,
                        prefix + txt + "Gamma": con.poly.matrix_uncond,
                        prefix + txt + "GLB": con.poly.lower_bound_uncond,
                        prefix + txt + "GUB": con.poly.upper_bound_uncond,
                    })
        # Write tensors to files
        for name, tensor in tensors.items():
            self.__tree.write_to_file_fp(name, tensor)

    def __str__(self):
        return (f"Problem Data\n"
                f"+ Variables: {(self.__num_states + self.__num_inputs) * self.__tree.num_nodes}\n"
                f"+ Step size: {self.__step_size}\n"
                f"+ Preconditioning: "
                f"requested = {self.__precondition_requested}, accepted = {self.__precondition_worth_it}")

    def __repr__(self):
        return (f"Problem Data\n+ Step size: {self.__step_size}\n"
                f"+ Preconditioning: "
                f"requested = {self.__precondition_requested}, accepted = {self.__precondition_worth_it}")


class Factory:
    """
    Risk-averse optimal control problem builder
    """

    def __init__(self, scenario_tree: tree.Tree, num_states, num_inputs):
        """
        :param scenario_tree: instance of ScenarioTree
        """
        self.__tree = scenario_tree
        self.__num_states = num_states
        self.__num_inputs = num_inputs
        self.__list_of_dynamics = [None for _ in range(self.__tree.num_nodes)]
        self.__nonleaf_cost = None
        self.__leaf_cost = None
        self.__nonleaf_constraint = None
        self.__leaf_constraint = None
        self.__list_of_risks = [None for _ in range(self.__tree.num_nonleaf_nodes)]
        self.__precondition = True
        self.__test = False
        self.__julia = False
        self.__preload_constraints()

    def __check_eventful(self, section):
        if not self.__tree.is_eventful:
            raise ValueError("Stochastic " + section + " provided but scenario tree made from data!")

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------
    def with_dynamics(self, dynamics):
        if dynamics.is_linear:
            self.__list_of_dynamics[0] = build.Dynamics(np.zeros(dynamics.A.shape),
                                                        np.zeros(dynamics.B.shape))
        if dynamics.is_affine:
            self.__list_of_dynamics[0] = build.Dynamics(np.zeros(dynamics.A.shape),
                                                        np.zeros(dynamics.B.shape),
                                                        np.zeros((dynamics.A.shape[0], 1)))
        for i in range(1, self.__tree.num_nodes):
            self.__list_of_dynamics[i] = deepcopy(dynamics)
        return self

    def with_dynamics_events(self, dynamics):
        self.__check_eventful("dynamics")
        temp_dyn = dynamics[0]
        if temp_dyn.is_linear:
            self.__list_of_dynamics[0] = build.Dynamics(np.zeros(temp_dyn.A.shape),
                                                        np.zeros(temp_dyn.B.shape))
        if dynamics[0].is_affine:
            self.__list_of_dynamics[0] = build.Dynamics(np.zeros(temp_dyn.A.shape),
                                                        np.zeros(temp_dyn.B.shape),
                                                        np.zeros((temp_dyn.A.shape[0], 1)))
        for i in range(1, self.__tree.num_nodes):
            event = self.__tree.event_of_node(i)
            self.__list_of_dynamics[i] = deepcopy(dynamics[event])
        return self

    def with_dynamics_list(self, dynamics):
        if dynamics[0] is not None:
            raise Exception(f"[ProblemFactory] First dynamics in list must be ({None})!")
        temp_dyn = dynamics[1]
        if temp_dyn.is_linear:
            self.__list_of_dynamics[0] = build.Dynamics(np.zeros(temp_dyn.A.shape),
                                                        np.zeros(temp_dyn.B.shape))
        if dynamics[1].is_affine:
            self.__list_of_dynamics[0] = build.Dynamics(np.zeros(temp_dyn.A.shape),
                                                        np.zeros(temp_dyn.B.shape),
                                                        np.zeros((temp_dyn.A.shape[0], 1)))
        for i in range(1, self.__tree.num_nodes):
            self.__list_of_dynamics[i] = deepcopy(dynamics[i])
        return self

    # --------------------------------------------------------
    # Costs
    # --------------------------------------------------------
    def with_cost_nonleaf(self, cost):
        list_ = [None for _ in range(self.__tree.num_nodes)]
        list_[0] = deepcopy(cost)
        list_[0].node_zero()
        for i in range(1, self.__tree.num_nodes):
            list_[i] = deepcopy(cost)
        self.__nonleaf_cost = cost.get_class()(list_)
        return self

    def with_cost_nonleaf_events(self, cost):
        self.__check_eventful("costs")
        list_ = [None for _ in range(self.__tree.num_nodes)]
        temp = cost[0]
        list_[0] = deepcopy(temp)
        list_[0].node_zero()
        for i in range(1, self.__tree.num_nodes):
            list_[i] = deepcopy(cost[self.__tree.event_of_node(i)])
        self.__nonleaf_cost = temp.get_class()(list_)
        return self

    def with_cost_nonleaf_list(self, cost):
        if cost[0] is not None:
            raise Exception(f"[ProblemFactory] First nonleaf cost in list must be ({None})!")
        cost[0] = deepcopy(cost[1])
        cost[0].node_zero()
        self.__nonleaf_cost = cost[0].get_class()(cost)
        return self

    def with_cost_leaf(self, cost):
        if not cost.leaf:
            raise Exception("[ProblemFactory] Cannot use nonleaf cost for leaf nodes!")
        list_ = [cost for _ in range(self.__tree.num_leaf_nodes)]
        self.__leaf_cost = cost.get_class()(list_)
        return self

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------
    def __preload_constraints(self):
        # load "No constraint"
        self.__nonleaf_constraint = build.No()
        self.__leaf_constraint = build.No()

    def with_constraint_nonleaf(self, state_input_constraint):
        self.__nonleaf_constraint = deepcopy(state_input_constraint)
        return self

    def with_constraint_leaf(self, state_constraint):
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
    # Preconditioning
    # --------------------------------------------------------
    def with_preconditioning(self, enable=True):
        self.__precondition = enable
        return self

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------
    def with_tests(self):
        self.__test = True
        return self

    # --------------------------------------------------------
    # Julia data
    # --------------------------------------------------------
    def with_julia(self):
        self.__julia = True
        return self

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------
    def generate_problem(self):
        """
        Generates problem data from the given build
        """
        problem = Problem(
            self.__tree,
            self.__num_states,
            self.__num_inputs,
            self.__list_of_dynamics,
            self.__nonleaf_cost,
            self.__leaf_cost,
            self.__nonleaf_constraint,
            self.__leaf_constraint,
            self.__list_of_risks,
            self.__precondition,
            self.__test,
            self.__julia,
        )
        return problem
