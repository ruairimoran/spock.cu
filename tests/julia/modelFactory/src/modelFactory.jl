module modelFactory

using JSON, JuMP, LinearAlgebra, MathOptInterface
const LA = LinearAlgebra
const MOI = MathOptInterface

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
Read a binary file.
Return: array of file data, array dimensions.
"""
function read_binary(::Type{T}, filename) where {T}
    dims = Array{UInt64}(undef, 3)
    data = Array{T}(undef, 0)
    open(filename, "r") do io
        # Read the first three UInt64 values for dimensions
        read!(io, dims)
        # Compute total number of elements
        num_el = prod(dims)
        # Read the remaining binary data
        data = Array{T}(undef, num_el)
        read!(io, data)
    end
    return data, dims
end

"""
Read a binary file into a tensor where:
1) The first three 64-bit unsigned integers (UInt64) represent the dimensions (m, n, k).
2) The remaining data are the tensor elements of type `T` (in column-major format).
Returns a vector of matrices of size ((m, n), k) with element type `T`.
"""
function read_tensor_from_binary(::Type{T}, filename) where {T}
    data, dims = read_binary(T, filename)
    # Reshape into a tensor
    arr = reshape(data, Int.(dims)...)
    return collect(eachslice(arr, dims = 3))
end

"""
Read a binary file into a vector where:
1) The first three 64-bit unsigned integers (UInt64) represent the dimensions (m, 1, 1).
2) The remaining data are the vector elements of type `T`.
Returns a vector of size (m, 1) with element type `T64`.
"""
function read_vector_from_binary(::Type{T}, filename) where {T}
    data, _ = read_binary(T, filename)
    # Reshape into a vector
    if T <: Int
        return Vector{Int64}(data)
    elseif T <: Real
        return Vector{Float64}(data)
    else
        throw("The type ($(T)) is not an int or real!")
    end
end

export read_array_from_binary, read_tensor_from_binary, read_vector_from_binary

"""
    Problem Data
"""
struct Data
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
    constraint_nonleaf_min :: Vector{TR}
    constraint_nonleaf_max :: Vector{TR}
    constraint_leaf_min :: Vector{TR}
    constraint_leaf_max :: Vector{TR}
    risk_type :: String
    risk_alpha :: TR
    risk_rowsS2 :: TI
    risk_rowsNNtr :: TI
    step_size :: TR
    ancestors :: Vector{TI}
    ch_num :: Vector{TI}
    ch_from :: Vector{TI}
    ch_to :: Vector{TI}
    conditional_probabilities :: Vector{TR}
end

export Data

"""
    Indexing
"""

node_to_x(d :: Data, node :: TI) = Vector{Int64}(collect(
    (node - 1) * d.num_states + 1 : node * d.num_states
))

node_to_u(d :: Data, node :: TI) = Vector{Int64}(collect(
    (node - 1) * d.num_inputs + 1 : node * d.num_inputs
))

node_to_ch(d :: Data, node :: TI) = Vector{Int64}(collect(
    d.ch_from[node] : d.chTo[node]
))

"""
    Build::Dynamics
"""

function impose_dynamics(
    model :: Model, 
    d :: Data, 
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
    d :: Data, 
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

"""
    Build::StateInputConstraints
"""

function impose_state_input_constraints(
    model :: Model, 
    d :: Data, 
    )
    x = model[:x]
    u = model[:u]

    if d.constraint_nonleaf != "rectangle" || d.constraint_leaf != "rectangle"
        throw("Constraint type not supported!")
    end

    nonleaf_x_min = d.constraint_nonleaf_min[1:d.num_states]
    nonleaf_x_max = d.constraint_nonleaf_max[1:d.num_states]
    nonleaf_u_min = d.constraint_nonleaf_min[d.num_states + 1:end]
    nonleaf_u_max = d.constraint_nonleaf_max[d.num_states + 1:end]

    @constraint(
        model,
        nonleaf_state_rectangle[node=2:d.num_nonleaf_nodes],
        nonleaf_x_min .<= x[node_to_x(d, node)] .<= nonleaf_x_max
    )

    @constraint(
        model,
        nonleaf_input_rectangle[node=1:d.num_nonleaf_nodes],
        nonleaf_u_min .<= u[node_to_u(d, node)] .<= nonleaf_u_max
    )

    @constraint(
        model,
        leaf_state_rectangle[node=d.num_nonleaf_nodes+1:d.num_nodes],
        d.constraint_leaf_min .<= x[node_to_x(d, node)] .<= d.constraint_leaf_max
    )
end

"""
    Build::Risk
"""

struct Risk
    dim :: TI
    E :: Matrix{TR}
    F :: Union{Nothing, Matrix{TR}}
    K :: MOI.AbstractSet
    b :: Vector{TR}

    function Risk(
        d :: Data, 
        node :: TI, 
        )
        if d.risk_type == "avar"
            alpha = d.risk_alpha
            ch_num = d.ch_num[node]
            ch_probs = d.conditional_probabilities[node_to_ch(node)]  # Conditional probabilities of children
            eye = I(ch_num)  # Identity matrix
            E = vcat(alpha * eye, -eye, ones(1, ch_num))
            F = nothing  # Matrix F is not applicable for AVaR
            K = MOI.CartesianProductCone(MOI.Nonnegatives(ch_num * 2), MOI.ZeroCone(1))
            b = vcat(ch_probs, zeros(ch_num), 1)
            dim = MOI.dimension(K)

            return new(dim, E, F, K, b)
        else
            throw("Risk type ($(d.risk_type)) not supported.")
        end
    end
end

export Risk

function impose_risk(
    model::Model, 
    d :: Data,
    risks :: Vector{Risk},
    )
    y = model[:y]
    t = model[:t]
    s = model[:s]

    y_idx = 0
    for node = 1:d.num_nonleaf_nodes
        dim = risks[node].dim
        y_node = y[y_idx + 1 : y_idx + dim]
        # y in K^*
        @constraint(
            model,
            in(y_node, MOI.dual_set(risks[node].K)))
        # y' * b <= s
        @constraint(
            model,
            y_node' * d.risks[node].b <= s[node]
        )
        # E' * y = t + s
        @constraint(
            model,
            risks[node].E' * y .== t[node_to_ch(node) .- 1] + s[node_to_ch(node)]
        )
#         # F' y = 0
#         @constraint(
#             model,
#             problem_definition.rms[i].F' * y[y_idx + 1 : y_idx + length(problem_definition.rms[i].b)] .== 0.
#         )
        # Add y dimension
        y_idx += dim
    end
end

"""
    Build::Model
"""

function build_model(
    solver,
    d :: Data,
    risks :: Vector{Risk},
    )
    model = Model(solver)
    set_silent(model)

    dim_y = 0
    for node = 1:d.num_nonleaf_nodes
        dim_y += risks[node].dim
    end

    @variable(model, x[i = 1 : d.num_nodes * d.num_states])
    @variable(model, u[i = 1 : d.num_nonleaf_nodes * d.num_inputs])
    @variable(model, y[i = 1 : dim_y])
    @variable(model, t[i = 1 : d.num_nodes - 1])
    @variable(model, s[i = 1 : d.num_nodes])

    @objective(model, Min, s[1])

    impose_dynamics(model, d)
    impose_cost(model, d)
    impose_state_input_constraints(model, d)
    impose_risk(model, d)

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
