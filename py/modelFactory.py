import numpy as np
import cvxpy as cp


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

    def solve(self, x0, solver=cp.SCS, tol=1e-3):
        self.__constraints.append(self.__x[self.__node_to_x(0)] == x0)
        self.__cvx = cp.Problem(self.__objective, self.__constraints)
        self.__constraints.pop()
        if solver == cp.MOSEK:
            return self.__cvx.solve(solver=solver,
                                    mosek_params={"MSK_DPAR_INTPNT_TOL_REL_GAP":tol,
                                                  "MSK_DPAR_INTPNT_CO_TOL_REL_GAP":tol,
                                                  "MSK_DPAR_INTPNT_QO_TOL_REL_GAP":tol
                                                  }
                                    )
        elif solver == cp.GUROBI:
            return self.__cvx.solve(solver=solver,
                                    FeasibilityTol=tol,
                                    OptimalityTol=tol)
        else:
            return self.__cvx.solve(solver=solver,
                                    eps=tol)

    @property
    def states(self):
        return np.array(self.__x.value).reshape(-1, 1)

    @property
    def inputs(self):
        return np.array(self.__u.value).reshape(-1, 1)

    @property
    def solve_time(self):
        return self.__cvx._solve_time

    def __build(self):
        """Build an optimisation model using CVXPY."""
        self.__x = cp.Variable((self.__tree.num_nodes * self.__problem.num_states,))
        self.__u = cp.Variable((self.__tree.num_nonleaf_nodes * self.__problem.num_inputs,))
        self.__y = cp.Variable((sum(self.__problem.risk_at_node(i).b.size
                                    for i in range(self.__tree.num_nonleaf_nodes))), )
        self.__t = cp.Variable((self.__tree.num_nodes - 1,))
        self.__s = cp.Variable((self.__tree.num_nodes,))
        self.__objective = cp.Minimize(self.__s[0])
        self.__impose_dynamics()
        self.__impose_cost()
        self.__impose_rectangle()
        self.__impose_risk_constraints()

    def __node_to_x(self, node):
        return slice(node * self.__nx, (node + 1) * self.__nx)

    def __node_to_u(self, node):
        return slice(node * self.__nu, (node + 1) * self.__nu)

    def __impose_dynamics(self):
        """Impose dynamic constraints on the optimisation model."""
        for node in range(1, self.__tree.num_nodes):
            anc = self.__tree.ancestor_of_node(node)
            print(self.__problem.dynamics_at_node(node).input.shape)
            print(self.__u[self.__node_to_u(anc)])
            self.__constraints.append(
                self.__x[self.__node_to_x(node)] ==
                self.__problem.dynamics_at_node(node).state @ self.__x[self.__node_to_x(anc)] +
                self.__problem.dynamics_at_node(node).input @ self.__u[self.__node_to_u(anc)] +
                self.__problem.dynamics_at_node(node).affine
            )

    def __impose_cost(self):
        """Impose cost constraints on the optimisation model."""
        if self.__problem.nonleaf_cost_at_node(1).is_linear:
            raise Exception("[Model] only supports quadratic nonleaf costs.\n")
        if self.__problem.leaf_cost_at_node(self.__tree.num_nonleaf_nodes).is_linear:
            raise Exception("[Model] only supports quadratic leaf costs.\n")
        # nonleaf
        for node in range(1, self.__tree.num_nodes):
            anc = self.__tree.ancestor_of_node(node)
            self.__constraints.append(
                cp.quad_form(self.__x[self.__node_to_x(anc)], self.__problem.nonleaf_cost_at_node(node).state) +
                cp.quad_form(self.__u[self.__node_to_u(anc)], self.__problem.nonleaf_cost_at_node(node).input)
                <= self.__t[node - 1]
            )
        # leaf
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            self.__constraints.append(
                cp.quad_form(self.__x[self.__node_to_x(node)], self.__problem.leaf_cost_at_node(node).state)
                <= self.__s[node]
            )

    def __impose_rectangle(self):
        """Impose box constraints on the variables."""
        if not self.__problem.nonleaf_constraint().is_rectangle:
            raise Exception("[Model] only supports rectangle nonleaf constraints.\n")
        if not self.__problem.leaf_constraint().is_rectangle:
            raise Exception("[Model] only supports rectangle leaf constraints.\n")
        nonleaf_lb_x = self.__problem.nonleaf_constraint().lower_bound[:self.__nx]
        nonleaf_ub_x = self.__problem.nonleaf_constraint().upper_bound[:self.__nx]
        nonleaf_lb_u = self.__problem.nonleaf_constraint().lower_bound[self.__nx:]
        nonleaf_ub_u = self.__problem.nonleaf_constraint().upper_bound[self.__nx:]
        for ele in range(self.__nu):
            self.__constraints.append(self.__u[self.__node_to_u(0)][ele] >= nonleaf_lb_u[ele])
            self.__constraints.append(self.__u[self.__node_to_u(0)][ele] <= nonleaf_ub_u[ele])
        for node in range(1, self.__tree.num_nonleaf_nodes):
            for ele in range(self.__nx):
                self.__constraints.append(self.__x[self.__node_to_x(node)][ele] >= nonleaf_lb_x[ele])
                self.__constraints.append(self.__x[self.__node_to_x(node)][ele] <= nonleaf_ub_x[ele])
            for ele in range(self.__nu):
                self.__constraints.append(self.__u[self.__node_to_u(node)][ele] >= nonleaf_lb_u[ele])
                self.__constraints.append(self.__u[self.__node_to_u(node)][ele] <= nonleaf_ub_u[ele])
        leaf_lb_x = self.__problem.leaf_constraint().lower_bound
        leaf_ub_x = self.__problem.leaf_constraint().upper_bound
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            for ele in range(self.__nx):
                self.__constraints.append(self.__x[self.__node_to_x(node)][ele] >= leaf_lb_x[ele])
                self.__constraints.append(self.__x[self.__node_to_x(node)][ele] <= leaf_ub_x[ele])

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
