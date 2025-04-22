import numpy as np
import cvxpy as cp
from copy import deepcopy


class Model:
    def __init__(self, tree, problem):
        self.__tree = tree
        self.__problem = problem
        self.__nx = self.__problem.num_states
        self.__nu = self.__problem.num_inputs
        self.__x0 = None
        self.__x = None
        self.__u = None
        self.__y = None
        self.__t = None
        self.__s = None
        self.__objective = None
        self.__constraints = []
        self.__cvx = None
        self.__build()

    def solve(self, x0, solver=cp.SCS, tol=1e-3, max_time=np.inf):
        # time limit in seconds
        self.__constraints.append(self.__x[self.__node_to_x(0)] == x0)
        self.__cvx = cp.Problem(self.__objective, self.__constraints)
        self.__constraints.pop()
        if solver == cp.MOSEK:
            mosek_params = {
                "MSK_DPAR_INTPNT_TOL_REL_GAP": tol,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
                "MSK_DPAR_INTPNT_QO_TOL_REL_GAP": tol,
                "MSK_DPAR_OPTIMIZER_MAX_TIME": max_time,
                "MSK_IPAR_LOG": 0,  # log = 3
                "MSK_IPAR_INFEAS_REPORT_AUTO": 0,  # log = 1
                "MSK_IPAR_INFEAS_REPORT_LEVEL": 0,  # log = 3
            }
            return self.__cvx.solve(solver=solver,
                                    mosek_params=mosek_params,
                                    verbose=False,
                                    )
        elif solver == cp.SCS:
            scs_params = {
                "eps_abs": tol,
                "eps_rel": tol,
                "eps_infeas": tol,
                "time_limit_secs": max_time,
            }
            return self.__cvx.solve(solver=solver,
                                    **scs_params,
                                    )
        else:
            return self.__cvx.solve(solver=solver,
                                    eps=tol,
                                    TimeLimit=max_time,
                                    )

    @property
    def states(self):
        return np.array(self.__x.value).reshape(-1, 1)

    @property
    def inputs(self):
        return np.array(self.__u.value).reshape(-1, 1)

    @property
    def solve_time(self):
        return self.__cvx.solver_stats.solve_time

    @property
    def status(self):
        return self.__cvx.status

    def __build(self):
        """Build an optimisation model using cp."""
        self.__x = cp.Variable((self.__tree.num_nodes * self.__problem.num_states,), name="x")
        self.__u = cp.Variable((self.__tree.num_nonleaf_nodes * self.__problem.num_inputs,), name="u")
        self.__y = cp.Variable((sum(self.__problem.risk_at_node(i).b.size
                                    for i in range(self.__tree.num_nonleaf_nodes))), name="y")
        self.__t = cp.Variable((self.__tree.num_nodes - 1,), name="t")
        self.__s = cp.Variable((self.__tree.num_nodes,), name="s")
        self.__objective = cp.Minimize(self.__s[0])
        self.__impose_dynamics()
        self.__impose_cost()
        self.__impose_constraints()
        self.__impose_risk_constraints()

    def __node_to_x(self, node):
        return slice(node * self.__nx, (node + 1) * self.__nx)

    def __node_to_u(self, node):
        return slice(node * self.__nu, (node + 1) * self.__nu)

    def __impose_dynamics(self):
        """Impose dynamic constraints on the optimisation model."""
        for node in range(1, self.__tree.num_nodes):
            if self.__problem.dynamics_at_node(node).is_affine:
                raise Exception("[Model] Affine dynamics not supported!")
            anc = self.__tree.ancestor_of_node(node)
            self.__constraints.append(
                self.__x[self.__node_to_x(node)] ==
                self.__problem.dynamics_at_node(node).A_uncond @ self.__x[self.__node_to_x(anc)] +
                self.__problem.dynamics_at_node(node).B_uncond @ self.__u[self.__node_to_u(anc)]
            )

    def __impose_cost(self):
        """Impose cost constraints on the optimisation model."""
        if self.__problem.nonleaf_cost().is_linear:
            raise Exception("[Model] only supports quadratic nonleaf costs.\n")
        if self.__problem.leaf_cost().is_linear:
            raise Exception("[Model] only supports quadratic leaf costs.\n")
        # nonleaf
        for node in range(1, self.__tree.num_nodes):
            anc = self.__tree.ancestor_of_node(node)
            self.__constraints.append(
                cp.quad_form(self.__x[self.__node_to_x(anc)], self.__problem.nonleaf_cost().Q_uncond[node]) +
                cp.quad_form(self.__u[self.__node_to_u(anc)], self.__problem.nonleaf_cost().R_uncond[node])
                <= self.__t[node - 1]
            )
        # leaf
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            leaf_idx = node - self.__tree.num_nonleaf_nodes
            self.__constraints.append(
                cp.quad_form(self.__x[self.__node_to_x(node)], self.__problem.leaf_cost().Q_uncond[leaf_idx])
                <= self.__s[node]
            )

    def __impose_rect_nonleaf(self, con):
        lb_x = con.lower_bound_uncond[:self.__nx]
        ub_x = con.upper_bound_uncond[:self.__nx]
        lb_u = con.lower_bound_uncond[self.__nx:]
        ub_u = con.upper_bound_uncond[self.__nx:]
        for ele in range(self.__nu):
            u_i = self.__u[self.__node_to_u(0)][ele]
            self.__constraints.append(u_i >= lb_u[ele])
            self.__constraints.append(u_i <= ub_u[ele])
        for node in range(1, self.__tree.num_nonleaf_nodes):
            for ele in range(self.__nx):
                x_i = self.__x[self.__node_to_x(node)][ele]
                self.__constraints.append(x_i >= lb_x[ele])
                self.__constraints.append(x_i <= ub_x[ele])
            for ele in range(self.__nu):
                u_i = self.__u[self.__node_to_u(node)][ele]
                self.__constraints.append(u_i >= lb_u[ele])
                self.__constraints.append(u_i <= ub_u[ele])

    def __impose_poly_nonleaf(self, con):
        lb = con.lower_bound_uncond
        ub = con.upper_bound_uncond
        g = con.matrix_uncond
        for node in range(self.__tree.num_nonleaf_nodes):
            x = self.__x[self.__node_to_x(node)].reshape((-1, 1), 'C')
            u = self.__u[self.__node_to_u(node)].reshape((-1, 1), 'C')
            xu = cp.vstack([x, u])
            v = g @ xu
            for ele in range(v.size):
                v_i = v[ele]
                self.__constraints.append(v_i >= lb[ele])
                self.__constraints.append(v_i <= ub[ele])

    def __impose_rect_leaf(self, con):
        lb_x = con.lower_bound_uncond
        ub_x = con.upper_bound_uncond
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            for ele in range(self.__nx):
                self.__constraints.append(self.__x[self.__node_to_x(node)][ele] >= lb_x[ele])
                self.__constraints.append(self.__x[self.__node_to_x(node)][ele] <= ub_x[ele])

    def __impose_poly_leaf(self, con):
        lb = con.lower_bound_uncond
        ub = con.upper_bound_uncond
        g = con.matrix_uncond
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            x = self.__x[self.__node_to_x(node)].reshape((-1, 1), 'C')
            v = g @ x
            for ele in range(v.size):
                v_i = v[ele]
                self.__constraints.append(v_i >= lb[ele])
                self.__constraints.append(v_i <= ub[ele])

    def __impose_constraints(self):
        """Impose convex constraints on the variables."""
        con = self.__problem.nonleaf_constraint()
        if con.is_rectangle:
            self.__impose_rect_nonleaf(con)
        elif con.is_polyhedron:
            self.__impose_poly_nonleaf(con)
        elif con.is_polyhedron_with_identity:
            self.__impose_rect_nonleaf(con.rect)
            self.__impose_poly_nonleaf(con.poly)
        con = self.__problem.leaf_constraint()
        if con.is_rectangle:
            self.__impose_rect_leaf(con)
        elif con.is_polyhedron:
            self.__impose_poly_leaf(con)
        elif con.is_polyhedron_with_identity:
            self.__impose_rect_leaf(con.rect)
            self.__impose_poly_leaf(con.poly)

    def __impose_risk_constraints(self):
        y_offset = 0
        for node in range(self.__tree.num_nonleaf_nodes):
            risk = self.__problem.risk_at_node(node)
            dim = risk.k
            y = self.__y[y_offset: y_offset + dim]
            # dual cone (y in K*)
            for i in range(dim - 1):
                self.__constraints.append(self.__y[y_offset + i] >= 0.)
            # y' * b <= s[i]
            self.__constraints.append(y.T @ risk.b <= self.__s[node])
            # E'y == t[ch] + s[ch] for children
            self.__constraints.append(risk.e.T @ y ==
                                      self.__t[self.__tree.children_of_node(node) - 1] +
                                      self.__s[self.__tree.children_of_node(node)])
            # F'y == 0
            # self.__constraints.append(risk.f.T @ y == 0)  # no F for AVaR
            y_offset += dim


class ModelWithPrecondition:
    def __init__(self, tree, problem):
        self.__tree = tree
        self.__problem = problem
        self.__nx = self.__problem.num_states
        self.__nu = self.__problem.num_inputs
        self.__x0 = None
        self.__x = None
        self.__u = None
        self.__y = None
        self.__t = None
        self.__s = None
        self.__objective = None
        self.__constraints = []
        self.__cvx = None
        self.__build()

    def solve(self, x0, solver=cp.SCS, tol=1e-3, max_time=np.inf):
        # time limit in seconds
        x0_cond = deepcopy(x0)
        for i in range(self.__problem.num_states):
            x0_cond[i] *= self.__problem.scaling[i]
        self.__constraints.append(self.__x[self.__node_to_x(0)] == x0_cond)
        self.__cvx = cp.Problem(self.__objective, self.__constraints)
        self.__constraints.pop()
        if solver == cp.MOSEK:
            mosek_params = {
                "MSK_DPAR_INTPNT_TOL_REL_GAP": tol,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
                "MSK_DPAR_INTPNT_QO_TOL_REL_GAP": tol,
                "MSK_DPAR_OPTIMIZER_MAX_TIME": max_time,
                "MSK_DPAR_MIO_MAX_TIME": max_time,
            }
            return self.__cvx.solve(solver=solver,
                                    mosek_params=mosek_params,
                                    )
        elif solver == cp.SCS:
            scs_params = {
                "eps_abs": tol,
                "eps_rel": tol,
                "eps_infeas": tol,
                "time_limit_secs": max_time,
            }
            return self.__cvx.solve(solver=solver,
                                    **scs_params,
                                    )
        else:
            return self.__cvx.solve(solver=solver,
                                    eps=tol,
                                    TimeLimit=max_time,
                                    )

    @property
    def states(self):
        states = np.array(self.__x.value).reshape(-1, 1)
        scaling = self.__problem.scaling[:self.__problem.num_states]
        for i in range(self.__tree.num_nodes * self.__problem.num_states):
            states[i] /= scaling[i % self.__problem.num_states]
        return states

    @property
    def inputs(self):
        inputs = np.array(self.__u.value).reshape(-1, 1)
        scaling = self.__problem.scaling[self.__problem.num_states:]
        for i in range(self.__tree.num_nonleaf_nodes * self.__problem.num_inputs):
            inputs[i] /= scaling[i % self.__problem.num_inputs]
        return inputs

    @property
    def solve_time(self):
        return self.__cvx.solve_time

    @property
    def status(self):
        return self.__cvx.status

    def __build(self):
        """Build an optimisation model using cp."""
        self.__x = cp.Variable((self.__tree.num_nodes * self.__problem.num_states,))
        self.__u = cp.Variable((self.__tree.num_nonleaf_nodes * self.__problem.num_inputs,))
        self.__y = cp.Variable((sum(self.__problem.risk_at_node(i).b.size
                                       for i in range(self.__tree.num_nonleaf_nodes))), )
        self.__t = cp.Variable((self.__tree.num_nodes - 1,))
        self.__s = cp.Variable((self.__tree.num_nodes,))
        self.__objective = cp.Minimize(self.__s[0])
        self.__impose_dynamics()
        self.__impose_cost()
        self.__impose_constraints()
        self.__impose_risk_constraints()

    def __node_to_x(self, node):
        return slice(node * self.__nx, (node + 1) * self.__nx)

    def __node_to_u(self, node):
        return slice(node * self.__nu, (node + 1) * self.__nu)

    def __impose_dynamics(self):
        """Impose dynamic constraints on the optimisation model."""
        for node in range(1, self.__tree.num_nodes):
            if self.__problem.dynamics_at_node(node).is_affine:
                raise Exception("[Model] Affine dynamics not supported!")
            anc = self.__tree.ancestor_of_node(node)
            self.__constraints.append(
                self.__x[self.__node_to_x(node)] ==
                self.__problem.dynamics_at_node(node).A @ self.__x[self.__node_to_x(anc)] +
                self.__problem.dynamics_at_node(node).B @ self.__u[self.__node_to_u(anc)]
            )

    def __impose_cost(self):
        """Impose cost constraints on the optimisation model."""
        if self.__problem.nonleaf_cost().is_linear:
            raise Exception("[Model] only supports quadratic nonleaf costs.\n")
        if self.__problem.leaf_cost().is_linear:
            raise Exception("[Model] only supports quadratic leaf costs.\n")
        # nonleaf
        for node in range(1, self.__tree.num_nodes):
            anc = self.__tree.ancestor_of_node(node)
            self.__constraints.append(
                cp.quad_form(self.__x[self.__node_to_x(anc)], self.__problem.nonleaf_cost().Q[node]) +
                cp.quad_form(self.__u[self.__node_to_u(anc)], self.__problem.nonleaf_cost().R[node])
                <= self.__t[node - 1]
            )

        # leaf
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            leaf_idx = node - self.__tree.num_nonleaf_nodes
            self.__constraints.append(
                cp.quad_form(self.__x[self.__node_to_x(node)], self.__problem.leaf_cost().Q[leaf_idx])
                <= self.__s[node]
            )

    def __impose_rect_nonleaf(self, con):
        lb_x = con.lower_bound[:self.__nx]
        ub_x = con.upper_bound[:self.__nx]
        lb_u = con.lower_bound[self.__nx:]
        ub_u = con.upper_bound[self.__nx:]
        for ele in range(self.__nu):
            u_i = self.__u[self.__node_to_u(0)][ele]
            self.__constraints.append(u_i >= lb_u[ele])
            self.__constraints.append(u_i <= ub_u[ele])
        for node in range(1, self.__tree.num_nonleaf_nodes):
            for ele in range(self.__nx):
                x_i = self.__x[self.__node_to_x(node)][ele]
                self.__constraints.append(x_i >= lb_x[ele])
                self.__constraints.append(x_i <= ub_x[ele])
            for ele in range(self.__nu):
                u_i = self.__u[self.__node_to_u(node)][ele]
                self.__constraints.append(u_i >= lb_u[ele])
                self.__constraints.append(u_i <= ub_u[ele])

    def __impose_poly_nonleaf(self, con):
        lb = con.lower_bound
        ub = con.upper_bound
        g = con.matrix
        for node in range(self.__tree.num_nonleaf_nodes):
            x = self.__x[self.__node_to_x(node)].reshape((-1, 1), 'C')
            u = self.__u[self.__node_to_u(node)].reshape((-1, 1), 'C')
            xu = cp.vstack([x, u])
            v = g @ xu
            for ele in range(v.size):
                v_i = v[ele]
                self.__constraints.append(v_i >= lb[ele])
                self.__constraints.append(v_i <= ub[ele])

    def __impose_rect_leaf(self, con):
        lb_x = con.lower_bound
        ub_x = con.upper_bound
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            for ele in range(self.__nx):
                self.__constraints.append(self.__x[self.__node_to_x(node)][ele] >= lb_x[ele])
                self.__constraints.append(self.__x[self.__node_to_x(node)][ele] <= ub_x[ele])

    def __impose_poly_leaf(self, con):
        lb = con.lower_bound
        ub = con.upper_bound
        g = con.matrix
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            x = self.__x[self.__node_to_x(node)].reshape((-1, 1), 'C')
            v = g @ x
            for ele in range(v.size):
                v_i = v[ele]
                self.__constraints.append(v_i >= lb[ele])
                self.__constraints.append(v_i <= ub[ele])

    def __impose_constraints(self):
        """Impose convex constraints on the variables."""
        con = self.__problem.nonleaf_constraint()
        if self.__problem.nonleaf_constraint().is_rectangle:
            self.__impose_rect_nonleaf(con)
        elif self.__problem.nonleaf_constraint().is_polyhedron:
            self.__impose_poly_nonleaf(con)
        elif self.__problem.nonleaf_constraint().is_polyhedron_with_identity:
            self.__impose_rect_nonleaf(con.rect)
            self.__impose_poly_nonleaf(con.poly)
        con = self.__problem.leaf_constraint()
        if self.__problem.leaf_constraint().is_rectangle:
            self.__impose_rect_leaf(con)
        elif self.__problem.leaf_constraint().is_polyhedron:
            self.__impose_poly_leaf(con)
        elif self.__problem.leaf_constraint().is_polyhedron_with_identity:
            self.__impose_rect_leaf(con.rect)
            self.__impose_poly_leaf(con.poly)

    def __impose_risk_constraints(self):
        y_offset = 0
        for node in range(self.__tree.num_nonleaf_nodes):
            risk = self.__problem.risk_at_node(node)
            dim = risk.k
            y = self.__y[y_offset: y_offset + dim]
            # dual cone (y in K*)
            for i in range(dim - 1):
                self.__constraints.append(self.__y[y_offset + i] >= 0.)
            # y' * b <= s[i]
            self.__constraints.append(y.T @ risk.b <= self.__s[node])
            # E'y == t[ch] + s[ch] for children
            self.__constraints.append(risk.e.T @ y ==
                                      self.__t[self.__tree.children_of_node(node) - 1] +
                                      self.__s[self.__tree.children_of_node(node)])
            # F'y == 0
            # self.__constraints.append(risk.f.T @ y == 0)  # no F for AVaR
            y_offset += dim
