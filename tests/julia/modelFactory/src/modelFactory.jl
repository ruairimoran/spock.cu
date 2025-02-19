module modelFactory

using JSON, JuMP

"""
    Filing system
"""
U = UInt64  # unsigned ints
T = Float64  # double precision
folder = "data/"
file_type = ".bt"
file_ext_u = "_u" * file_type
if T == Float32
    file_ext_t = "_f" * file_type
else
    file_ext_t = "_d" * file_type
end

export U, T, folder, file_ext_u, file_ext_t

"""
Read a binary file where:
1) The first three 64-bit unsigned integers (UInt64) represent the dimensions (m, n, k).
2) The remaining data are the array elements of type `T` (in column-major format).
Returns a tensor of size (m, n, k) with element type `T`.
"""
function read_tensor_from_binary(::Type{T}, filename) where {T}
    open(filename, "r") do io
        # Read the first three UInt64 values for dimensions
        dims = Array{UInt64}(undef, 3)
        read!(io, dims)
        # Compute total number of elements
        num_el = prod(dims)
        # Read the remaining binary data as a Vector{T}
        data = Array{T}(undef, num_el)
        read!(io, data)
        # Reshape into a tensor
        return reshape(data, Int.(dims)...)
    end
end

export read_tensor_from_binary

"""
    Json::DataStructure
"""
struct JSON_DATA
    num_events :: UInt
    num_nonleaf_nodes :: UInt
    num_nodes :: UInt
    num_stages :: UInt
    num_states :: UInt
    num_inputs :: UInt
    dynamics_type :: String
    dynamics_A :: Array
    dynamics_B :: Array
    dynamics_c :: Array
    constraint_nonleaf :: String
    constraint_leaf :: String
    risk_type :: String
    risk_alpha :: Real
    risk_rowsS2 :: UInt
    risk_rowsNNtr :: UInt
    step_size :: Real
    ancestors :: Array
end

"""
    Indexing
"""

function node_to_x(data :: JSON_DATA, node :: UInt)
    return collect(
        (node - 1) * data.num_states + 1 : node * data.num_states
    )
end

function node_to_u(data :: JSON_DATA, node :: UInt)
    return collect(
        (node - 1) * data.num_inputs + 1 : node * data.num_inputs
    )
end

"""
    Build::Dynamics
"""

function impose_dynamics(
    model :: Model,
    d :: JSON_DATA,
    )
    x = model[:x]
    u = model[:u]

    @constraint(
    model,
    dynamics[node=2:d.num_nodes, row=1:d.num_states],
    x[node_to_x(d, node)[row]] ==
        sum(d.dynamics_A[row, col, node] * x[node_to_x(d, d.ancestors[node])[col]] for col in 1:d.num_states)
        + sum(d.dynamics_B[row, col, node] * u[node_to_u(d, d.ancestors[node])[col]] for col in 1:d.num_inputs)
        + d.dynamics_c[row, 1, node]
    )
end

# """
#     Build::Cost
# """

# function impose_cost(
#     model :: Model,
#     data :: JSON_DATA,
#     )
#     x = model[:x]
#     u = model[:u]
#     t = model[:t]
#     s = model[:s]

#     @constraint(
#     model,
#     nonleaf_cost[node = 2:problem_definition.scen_tree.n],
#     (x[node_to_x(problem_definition, anc_mapping[node])]' * problem_definition.cost.Q[node - 1] * x[node_to_x(problem_definition, anc_mapping[node])] + u[node_to_u(problem_definition, anc_mapping[node])]' * problem_definition.cost.R[node - 1] * u[node_to_u(problem_definition, anc_mapping[node])]) <= model[:tau][node - 1]
#     )

#     @constraint(
#     model,
#     leaf_cost[i = problem_definition.scen_tree.leaf_node_min_index : problem_definition.scen_tree.leaf_node_max_index],
#     x[node_to_x(problem_definition, i)]' * problem_definition.cost.QN[i - problem_definition.scen_tree.leaf_node_min_index + 1] * x[node_to_x(problem_definition, i)] <= model[:s][i]
#     )
# end

# """
#     Build::StateInputConstraints
# """
#
# function impose_state_input_constraints(
#     model :: Model,
#     problem_definition :: GENERIC_PROBLEM_DEFINITION,
#     )
#     x = model[:x]
#     u = model[:u]
#
#     x_min = problem_definition.constraints.x_min
#     x_max = problem_definition.constraints.x_max
#     u_min = problem_definition.constraints.u_min
#     u_max = problem_definition.constraints.u_max
#
#     @constraint(
#     model,
#     x_box_min[i=1:problem_definition.scen_tree.n],
#     x[node_to_x(problem_definition, i)] .>= x_min
#     )
#
#     @constraint(
#     model,
#     x_box_max[i=1:problem_definition.scen_tree.n],
#     x[node_to_x(problem_definition, i)] .<= x_max
#     )
#
#     @constraint(
#     model,
#     u_box_min[i=1:problem_definition.scen_tree.n_non_leaf_nodes],
#     u[node_to_u(problem_definition, i)] .>= u_min
#     )
#
#     @constraint(
#     model,
#     u_box_max[i=1:problem_definition.scen_tree.n_non_leaf_nodes],
#     u[node_to_u(problem_definition, i)] .<= u_max
#     )
# end
#
# """
#     Build::Risk
# """
#
# function add_risk_epi_constraints(
#     model::Model,
#     problem_definition :: GENERIC_PROBLEM_DEFINITION,
#     )
#     ny = 0
#     for i = 1:length(problem_definition.rms)
#     ny += length(problem_definition.rms[i].b)
#     end
#     @variable(model, y[i=1:ny])
#
#     y = model[:y]
#     s = model[:s]
#     tau = model[:tau]
#
#     y_offset = 0
#     for i = 1:problem_definition.scen_tree.n_non_leaf_nodes
#         # y in K^*
#         dim_offset = 0
#         for j = 1:1#length(problem_definition.rms[i].K.subcones)
#           dim = MOI.dimension(problem_definition.rms[i].K.subcones[j])
#           @constraint(model, in(y[y_offset + dim_offset + 1 : y_offset + dim_offset + dim], MOI.dual_set(problem_definition.rms[i].K.subcones[j])))
#           dim_offset += dim
#         end
#
#         # y' * b <= s
#         @constraint(
#           model,
#           y[y_offset + 1 : y_offset + length(problem_definition.rms[i].b)]' * problem_definition.rms[i].b <= s[i]
#         )
#
#
#         # E^i' * y = tau + s
#         for j in problem_definition.scen_tree.child_mapping[i]
#           @constraint(
#             model,
#             problem_definition.rms[i].E' * y[y_offset + 1 : y_offset + length(problem_definition.rms[i].b)] .== tau[j - 1] + s[j]
#           )
#         end
#
#
#         # F' y = 0
#         @constraint(
#           model,
#           problem_definition.rms[i].F' * y[y_offset + 1 : y_offset + length(problem_definition.rms[i].b)] .== 0.
#         )
#
#
#         y_offset += length(problem_definition.rms[i].b)
#     end
# end

"""
    Build::Model
"""

function build_model(
    solver,
    )

    json = JSON.parse(read(folder * "data.json", String))
    data = JSON_DATA(
        json["numEvents"],
        json["numNonleafNodes"],
        json["numNodes"],
        json["numStages"],
        json["numStates"],
        json["numInputs"],
        json["dynamics"]["type"],
        read_tensor_from_binary(T, folder * "dynamics_A" * file_ext_t),
        read_tensor_from_binary(T, folder * "dynamics_B" * file_ext_t),
        read_tensor_from_binary(T, folder * "dynamics_e" * file_ext_t),
        json["constraint"]["nonleaf"],
        json["constraint"]["leaf"],
        json["risk"]["type"],
        json["risk"]["alpha"],
        json["rowsS2"],
        json["rowsNNtr"],
        json["stepSize"],
        read_tensor_from_binary(U, folder * "ancestors" * file_ext_u) .+ 1,
    )

    model = Model(solver)
    set_silent(model)

    @variable(model, x[i=1:data.num_nodes * data.num_states])
    @variable(model, u[i=1:data.num_nonleaf_nodes * data.num_inputs])
    @variable(model, t[i=1:data.num_nodes - 1])
    @variable(model, s[i=1:data.num_nodes])

    @objective(model, Min, s[1])

    impose_dynamics(model, data)
#     impose_cost(model, problem_definition)
#     impose_state_input_constraints(model, problem_definition)
#     add_risk_epi_constraints(model, problem_definition)

    return model

end

export build_model

# """
#     JuMP::Solve
# """
# function solve(model :: REFERENCE_MODEL, x0 :: AbstractArray{TF, 1}) where {TF <: Real}
#     # Add initial state constraint
# #     @constraint(model, initial_condition[i=1:length(x0)], model[:x][i] .== x0[i])
# #     # Solve problem
# #     optimize!(model)
# #     # Return states and inputs
# #     return value.(model[:x]), value.(model[:u])
# end
#
# export solve

end  # End of module
