module modelFactory

using JSON, JuMP, LinearAlgebra, MathOptInterface
const LA = LinearAlgebra
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

"""
    Filing system
"""
const TI = Int64  # ints
const TR = Float64  # double
const folder = "data/"
const file_type = ".bt"
const file_ext_i = "_u" * file_type
if TR == Float32
    const file_ext_r = "_f" * file_type
else
    const file_ext_r = "_d" * file_type
end
const minute = 60

export TI, TR, folder, file_ext_r, minute

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

export read_vector_from_binary

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

function read_data()
    json = JSON.parse(read(folder * "data.json", String))
    return Data(
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
        read_vector_from_binary(TR, folder * "nonleafConstraintILB" * file_ext_r),
        read_vector_from_binary(TR, folder * "nonleafConstraintIUB" * file_ext_r),
        read_vector_from_binary(TR, folder * "leafConstraintILB" * file_ext_r),
        read_vector_from_binary(TR, folder * "leafConstraintIUB" * file_ext_r),
        json["risk"]["type"],
        json["risk"]["alpha"],
        json["rowsS2"],
        json["rowsNNtr"],
        json["stepSize"],
        read_vector_from_binary(TI, folder * "ancestors" * file_ext_i) .+ 1,
        read_vector_from_binary(TI, folder * "numChildren" * file_ext_i),
        read_vector_from_binary(TI, folder * "childrenFrom" * file_ext_i) .+ 1,
        read_vector_from_binary(TI, folder * "childrenTo" * file_ext_i) .+ 1,
        read_vector_from_binary(TR, folder * "conditionalProbabilities" * file_ext_r),
    )
end

export read_data

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
    d.ch_from[node] : d.ch_to[node]
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
        <= t[node - 1]
    )

    @constraint(
        model,
        leaf_cost[node=d.num_nonleaf_nodes+1:d.num_nodes],
        x[node_to_x(d, node)]' * d.cost_leaf_Q[node - d.num_nonleaf_nodes] * x[node_to_x(d, node)]
        <= s[node]
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

@enum RiskType AVaR

struct Risk
    type :: RiskType
    dim :: TI
    E :: Matrix{TR}
    F :: Union{Nothing, Matrix{TR}}
    b :: Vector{TR}

    function Risk(
        d :: Data, 
        node :: TI, 
        )
        if d.risk_type == "avar"
            type = AVaR
        else
            throw("Risk type ($(d.risk_type)) not supported!")
        end

        if type == AVaR
            alpha = d.risk_alpha
            ch_num = d.ch_num[node]
            ch_probs = d.conditional_probabilities[node_to_ch(d, node)]  # Conditional probabilities of children
            eye = I(ch_num)  # Identity matrix
            dim = ch_num * 2 + 1
            E = vcat(alpha * eye, -eye, ones(1, ch_num))
            F = nothing  # Matrix F is not applicable for AVaR
            b = vcat(ch_probs, zeros(ch_num), 1)
            return new(type, dim, E, F, b)
        else
            throw("Risk type not implemented!")
        end
    end
end

function build_risk(
    d :: Data,
    )
    return [Risk(d, node) for node in 1:d.num_nonleaf_nodes]
end

export build_risk

function risk_add_dual_cone_constraint(
    this :: Risk,
    model :: Model,
    y :: Vector{VariableRef},
    )
    if this.type == AVaR
        @constraint(
        model,
        in(y[1:this.dim-1], MOI.Nonnegatives(this.dim - 1))
        )
        # No need to project y[dim] on Reals
    else
        throw("Risk type not implemented!")
    end
end

function impose_risk(
    model :: Model, 
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
        # y in K*
        risk_add_dual_cone_constraint(risks[node], model, y_node)
        # y' * b <= s
        @constraint(
            model,
            y_node' * risks[node].b <= s[node]
        )
        # E' * y = t[ch] + s[ch]
        @constraint(
            model,
            risks[node].E' * y_node .== t[node_to_ch(d, node) .- 1] + s[node_to_ch(d, node)]
        )
        # F' y = 0
        # @constraint(
        #     model,
        #     risks[node].F' * y_node .== 0.
        # )
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
    risk :: Vector{Risk},
    )
    model = Model(solver)
    set_silent(model)

    dim_y = 0
    for node = 1:d.num_nonleaf_nodes
        dim_y += risk[node].dim
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
    impose_risk(model, d, risk)

    return model

end

export build_model

"""
    JuMP::Solve
"""

function solve_this(
    model :: Model, 
    x0 :: Vector{TR},
    )
    x = model[:x]
    u = model[:u]
    # Add initial state constraint
    @constraint(
        model, 
        initial_condition[element = 1 : length(x0)], 
        x[element] .== x0[element]
    )
    # Solve problem
    optimize!(model)
    # Return states and inputs
    return value.(x), value.(u)
end

export solve_this

end  # End of module
