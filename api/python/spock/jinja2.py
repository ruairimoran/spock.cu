{
\# TEMPLATE FILE FOR JSON DATA #\
\# total number of events #\
"numEvents": \~num_events~\,
\# total number of nonleaf nodes #\
"numNonleafNodes": \~num_nonleaf_nodes~\,
\# total number of nodes #\
"numNodes": \~num_nodes~\,
\# total number of stages (N+1) #\
"numStages": \~num_stages~\,
\# NUMBER OF STATES #\
"numStates": \~num_states~\,
\# NUMBER OF INPUTS #\
"numInputs": \~num_inputs~\,
\# DYNAMICS #\
"dynamics": {"type": \% if dynamics.is_linear %\"linear"\% elif dynamics.is_affine %\"affine"\% endif %\
},
\# COSTS #\
"cost": {"nonleaf": \% if nonleaf_cost.is_quadratic %\"quadratic"\% elif nonleaf_cost.is_linear %\"linear"\% elif nonleaf_cost.is_quadratic_plus_linear %\"quadraticPlusLinear"\% endif
%\, "leaf": \% if leaf_cost.is_quadratic %\"quadratic"\% elif leaf_cost.is_linear %\"linear"\% elif leaf_cost.is_quadratic_plus_linear %\"quadraticPlusLinear"\% endif %\
},
\# CONSTRAINTS #\
"constraint": {"nonleaf": \% if nonleaf_constraint.is_no %\"no"\% elif nonleaf_constraint.is_rectangle %\"rectangle"\% elif nonleaf_constraint.is_polyhedron %\"polyhedron"\% elif nonleaf_constraint.is_polyhedron_with_identity %\"polyhedronWithIdentity"\% endif
%\, "leaf": \% if leaf_constraint.is_no %\"no"\% elif leaf_constraint.is_rectangle %\"rectangle"\% elif leaf_constraint.is_polyhedron %\"polyhedron"\% elif leaf_constraint.is_polyhedron_with_identity %\"polyhedronWithIdentity"\% endif %\
},
\# RISKS #\
"risk": {"type": \% if risk.is_avar %\"avar", "alpha": \~risk.alpha~\\% elif risk.is_evar %\"evar"\% endif %\
},
"rowsS2": \~ker_con_rows~\,
"rowsNNtr": \~null_dim~\,
"stepSize": \~step_size~\,
"preconditioned": \% if precondition %\true\% else %\false\% endif %\

}