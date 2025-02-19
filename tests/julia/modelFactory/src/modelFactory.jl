module modelFactory

using JSON, JuMP

"""
    Filing system
"""
TI = Int64  # ints
TR = Float64  # double
folder = "data/"
file_type = ".bt"
file_ext_i = "_u" * file_type
if TR == Float32
    file_ext_r = "_f" * file_type
else
    file_ext_r = "_d" * file_type
end

export TI, TR, folder, file_ext_i, file_ext_r

"""
Read a binary file into an array where:
1) The first three 64-bit unsigned integers (UInt64) represent the dimensions (m, n, k).
2) The remaining data are the array elements of type `T` (in column-major format).
Returns an array of size (m, n, k) with element type `T`.
"""
function read_array_from_binary(::Type{T}, filename) where {T}
    open(filename, "r") do io
        # Read the first three UInt64 values for dimensions
        dims = Array{UInt64}(undef, 3)
        read!(io, dims)
        # Compute total number of elements
        num_el = prod(dims)
        # Read the remaining binary data
        data = Array{T}(undef, num_el)
        read!(io, data)
        # Reshape into an array
        return reshape(data, Int.(dims)...)
    end
end

"""
Read a binary file into a tensor where:
1) The first three 64-bit unsigned integers (UInt64) represent the dimensions (m, n, k).
2) The remaining data are the tensor elements of type `T` (in column-major format).
Returns a vector of matrices of size ((m, n), k) with element type `T`.
"""
function read_tensor_from_binary(::Type{T}, filename) where {T}
    open(filename, "r") do io
        # Read the first three UInt64 values for dimensions
        dims = Array{UInt64}(undef, 3)
        read!(io, dims)
        # Compute total number of elements
        num_el = prod(dims)
        # Read the remaining binary data
        data = Array{T}(undef, num_el)
        read!(io, data)
        # Reshape into a tensor
        arr = reshape(data, Int.(dims)...)
        return collect(eachslice(arr, dims=3))
    end
end

export read_array_from_binary, read_tensor_from_binary

"""
    Json::DataStructure
"""
struct JSON_DATA
    num_events :: TI
    num_nonleaf_nodes :: TI
    num_nodes :: TI
    num_stages :: TI
    num_states :: TI
    num_inputs :: TI
    dynamics_type :: String
    dynamics_A :: Vector{Matrix{TR}}
    dynamics_B :: Vector{Matrix{TR}}
    dynamics_c :: Vector{Matrix{TR}}
    cost_nonleaf_Q :: Vector{Matrix{TR}}
    cost_nonleaf_R :: Vector{Matrix{TR}}
    cost_leaf_Q :: Vector{Matrix{TR}}
    constraint_nonleaf :: String
    constraint_leaf :: String
    risk_type :: String
    risk_alpha :: TR
    risk_rowsS2 :: TI
    risk_rowsNNtr :: TI
    step_size :: TR
    ancestors :: Array{TI}
end

"""
    Indexing
"""

function node_to_x(data :: JSON_DATA, node :: TI)
    return Vector{Int64}(collect(
        (node - 1) * data.num_states + 1 : node * data.num_states
    ))
end

function node_to_u(data :: JSON_DATA, node :: TI)
    return Vector{Int64}(collect(
        (node - 1) * data.num_inputs + 1 : node * data.num_inputs
    ))
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
    dynamics[node=2:d.num_nodes],
    x[node_to_x(d, node)] .==
        d.dynamics_A[node] * x[node_to_x(d, d.ancestors[node])]
        + d.dynamics_B[node] * u[node_to_u(d, d.ancestors[node])]
        + d.dynamics_c[node]
    )
end

"""
    Build::Cost
"""

function impose_cost(
    model :: Model,
    d :: JSON_DATA,
    )
    x = model[:x]
    u = model[:u]
    t = model[:t]
    s = model[:s]

    @constraint(
    model,
    nonleaf_cost[node=2:d.num_nodes],
    (x[node_to_x(d, d.ancestors[node])]' * d.cost_nonleaf_Q[node] * x[node_to_x(d, d.ancestors[node])] 
    + u[node_to_u(d, d.ancestors[node])]' * d.cost_nonleaf_R[node] * u[node_to_u(d, d.ancestors[node])]) 
    <= model[:t][node - 1]
    )

    @constraint(
    model,
    leaf_cost[node=d.num_nonleaf_nodes+1:d.num_nodes],
    x[node_to_x(d, node)]' * d.cost_leaf_Q[node - d.num_nonleaf_nodes] * x[node_to_x(d, node)] 
    <= model[:s][node]
    )
end

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
        read_tensor_from_binary(TR, folder * "dynamics_A" * file_ext_r),
        read_tensor_from_binary(TR, folder * "dynamics_B" * file_ext_r),
        read_tensor_from_binary(TR, folder * "dynamics_e" * file_ext_r),
        read_tensor_from_binary(TR, folder * "cost_nonleafQ" * file_ext_r),
        read_tensor_from_binary(TR, folder * "cost_nonleafR" * file_ext_r),
        read_tensor_from_binary(TR, folder * "cost_leafQ" * file_ext_r),
        json["constraint"]["nonleaf"],
        json["constraint"]["leaf"],
        json["risk"]["type"],
        json["risk"]["alpha"],
        json["rowsS2"],
        json["rowsNNtr"],
        json["stepSize"],
        read_array_from_binary(TI, folder * "ancestors" * file_ext_i) .+ 1,
    )

    model = Model(solver)
    set_silent(model)

    @variable(model, x[i=1:data.num_nodes * data.num_states])
    @variable(model, u[i=1:data.num_nonleaf_nodes * data.num_inputs])
    @variable(model, t[i=1:data.num_nodes - 1])
    @variable(model, s[i=1:data.num_nodes])

    @objective(model, Min, s[1])

    impose_dynamics(model, data)
    impose_cost(model, data)
    # impose_state_input_constraints(model, problem_definition)
    # add_risk_epi_constraints(model, problem_definition)

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
