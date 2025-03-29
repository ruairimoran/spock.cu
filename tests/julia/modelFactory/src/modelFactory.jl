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
    tensor = [copy(A) for A in eachslice(arr, dims=3)]
    if dims[3] == 1
        return first(tensor)
    end
    return tensor
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
    cost_nonleaf_q :: Union{Vector{Matrix{TR}}, Nothing}
    cost_nonleaf_r :: Union{Vector{Matrix{TR}}, Nothing}
    cost_leaf_Q :: Vector{Matrix{TR}}
    constraint_nonleaf :: String
    constraint_leaf :: String
    constraint_nonleaf_ilb :: Union{Vector{TR}, Nothing}
    constraint_nonleaf_iub :: Union{Vector{TR}, Nothing}
    constraint_nonleaf_glb :: Union{Vector{TR}, Nothing}
    constraint_nonleaf_gub :: Union{Vector{TR}, Nothing}
    constraint_nonleaf_g :: Union{Matrix{TR}, Nothing}
    constraint_leaf_ilb :: Union{Vector{TR}, Nothing}
    constraint_leaf_iub :: Union{Vector{TR}, Nothing}
    constraint_leaf_glb :: Union{Vector{TR}, Nothing}
    constraint_leaf_gub :: Union{Vector{TR}, Nothing}
    constraint_leaf_g :: Union{Matrix{TR}, Nothing}
    risk_type :: String
    risk_alpha :: TR
    ancestors :: Vector{TI}
    ch_num :: Vector{TI}
    ch_from :: Vector{TI}
    ch_to :: Vector{TI}
    conditional_probabilities :: Vector{TR}
end

function read_data()
    json = JSON.parse(read(folder * "data.json", String))
    if false
        cost_nonleaf_q = read_tensor_from_binary(TR, folder * "uncond_cost_nonleaf_q" * file_ext_r)
    else
        cost_nonleaf_q = nothing
    end
    if true
        cost_nonleaf_r = read_tensor_from_binary(TR, folder * "uncond_cost_nonleaf_r" * file_ext_r)
    else
        cost_nonleaf_r = nothing
    end    
    constraint_nonleaf = json["constraint"]["nonleaf"]
    constraint_leaf = json["constraint"]["leaf"]
    constraint_nonleaf_ilb = nothing
    constraint_nonleaf_iub = nothing
    constraint_nonleaf_glb = nothing
    constraint_nonleaf_gub = nothing
    constraint_nonleaf_g = nothing
    constraint_leaf_ilb = nothing
    constraint_leaf_iub = nothing
    constraint_leaf_glb = nothing
    constraint_leaf_gub = nothing
    constraint_leaf_g = nothing
    if constraint_nonleaf == "rectangle" || constraint_nonleaf == "polyhedronWithIdentity"
        constraint_nonleaf_ilb = read_vector_from_binary(TR, folder * "uncond_nonleafConstraintILB" * file_ext_r)
        constraint_nonleaf_iub = read_vector_from_binary(TR, folder * "uncond_nonleafConstraintIUB" * file_ext_r)
    end
    if constraint_nonleaf == "polyhedron" || constraint_nonleaf == "polyhedronWithIdentity"
        constraint_nonleaf_glb = read_vector_from_binary(TR, folder * "uncond_nonleafConstraintGLB" * file_ext_r)
        constraint_nonleaf_gub = read_vector_from_binary(TR, folder * "uncond_nonleafConstraintGUB" * file_ext_r)
        constraint_nonleaf_g = read_tensor_from_binary(TR, folder * "uncond_nonleafConstraintGamma" * file_ext_r)
    end
    if constraint_leaf == "rectangle" || constraint_leaf == "polyhedronWithIdentity"
        constraint_leaf_ilb = read_vector_from_binary(TR, folder * "uncond_leafConstraintILB" * file_ext_r)
        constraint_leaf_iub = read_vector_from_binary(TR, folder * "uncond_leafConstraintIUB" * file_ext_r)
    end
    if constraint_leaf == "polyhedron" || constraint_leaf == "polyhedronWithIdentity"
        constraint_leaf_glb = read_vector_from_binary(TR, folder * "uncond_leafConstraintGLB" * file_ext_r)
        constraint_leaf_gub = read_vector_from_binary(TR, folder * "uncond_leafConstraintGUB" * file_ext_r)
        constraint_leaf_g = read_tensor_from_binary(TR, folder * "uncond_leafConstraintGamma" * file_ext_r)
    end
    data = Data(
        json["numEvents"],
        json["numNonleafNodes"],
        json["numNodes"],
        json["numStages"],
        json["numStates"],
        json["numInputs"],
        json["dynamics"]["type"],
        read_tensor_from_binary(TR, folder * "uncond_dynamics_A" * file_ext_r),
        read_tensor_from_binary(TR, folder * "uncond_dynamics_B" * file_ext_r),
        read_tensor_from_binary(TR, folder * "uncond_dynamics_c" * file_ext_r),
        read_tensor_from_binary(TR, folder * "uncond_cost_nonleaf_Q" * file_ext_r),
        read_tensor_from_binary(TR, folder * "uncond_cost_nonleaf_R" * file_ext_r),
        cost_nonleaf_q,
        cost_nonleaf_r,
        read_tensor_from_binary(TR, folder * "uncond_cost_leaf_Q" * file_ext_r),
        constraint_nonleaf,
        constraint_leaf,
        constraint_nonleaf_ilb,
        constraint_nonleaf_iub,
        constraint_nonleaf_glb,
        constraint_nonleaf_gub,
        constraint_nonleaf_g,
        constraint_leaf_ilb,
        constraint_leaf_iub,
        constraint_leaf_glb,
        constraint_leaf_gub,
        constraint_leaf_g,
        json["risk"]["type"],
        json["risk"]["alpha"],
        read_vector_from_binary(TI, folder * "ancestors" * file_ext_i) .+ 1,
        read_vector_from_binary(TI, folder * "numChildren" * file_ext_i),
        read_vector_from_binary(TI, folder * "childrenFrom" * file_ext_i) .+ 1,
        read_vector_from_binary(TI, folder * "childrenTo" * file_ext_i) .+ 1,
        read_vector_from_binary(TR, folder * "conditionalProbabilities" * file_ext_r),
    )
    return data
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
    if true
        println("[JuMP] Costs have been modified for power example!")
        @constraint(
            model,
            nonleaf_cost[node=2:d.num_nodes],
            (
                x[node_to_x(d, d.ancestors[node])]' * d.cost_nonleaf_Q[node] * x[node_to_x(d, d.ancestors[node])]
                +
                u[node_to_u(d, d.ancestors[node])]' * d.cost_nonleaf_R[node] * u[node_to_u(d, d.ancestors[node])]
                +
                (d.cost_nonleaf_r[node]' * u[node_to_u(d, d.ancestors[node])])[1]
            )
            <= t[node - 1]
        )
    else
        @constraint(
            model,
            nonleaf_cost[node=2:d.num_nodes],
            (
                x[node_to_x(d, d.ancestors[node])]' * d.cost_nonleaf_Q[node] * x[node_to_x(d, d.ancestors[node])]
                +
                u[node_to_u(d, d.ancestors[node])]' * d.cost_nonleaf_R[node] * u[node_to_u(d, d.ancestors[node])]
            )
            <= t[node - 1]
        )
    end
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

function impose_rect_nonleaf(model :: Model, d :: Data)
    x = model[:x]
    u = model[:u]
    x_min = d.constraint_nonleaf_ilb[1:d.num_states]
    x_max = d.constraint_nonleaf_iub[1:d.num_states]
    u_min = d.constraint_nonleaf_ilb[d.num_states + 1:end]
    u_max = d.constraint_nonleaf_iub[d.num_states + 1:end]
    @constraint(
        model,
        nonleaf_rect_state[node=2:d.num_nonleaf_nodes],
        x_min .<= x[node_to_x(d, node)] .<= x_max
    )
    @constraint(
        model,
        nonleaf_rect_input[node=1:d.num_nonleaf_nodes],
        u_min .<= u[node_to_u(d, node)] .<= u_max
    )
end

function impose_poly_nonleaf(model :: Model, d :: Data)
    x = model[:x]
    u = model[:u]
    g_min = d.constraint_nonleaf_glb
    g_max = d.constraint_nonleaf_gub
    g = d.constraint_nonleaf_g
    @constraint(
        model,
        nonleaf_poly[node=1:d.num_nonleaf_nodes],
        g_min .<= g * vcat(x[node_to_x(d, node)], u[node_to_u(d, node)]) .<= g_max
    )
end

function impose_rect_leaf(model :: Model, d :: Data)
    x = model[:x]
    u = model[:u]
    x_min = d.constraint_leaf_ilb
    x_max = d.constraint_leaf_iub
    @constraint(
        model,
        leaf_rect[node=d.num_nonleaf_nodes+1:d.num_nodes],
        x_min .<= x[node_to_x(d, node)] .<= x_max
    )
end

function impose_poly_leaf(model :: Model, d :: Data)
    x = model[:x]
    g_min = d.constraint_leaf_glb
    g_max = d.constraint_leaf_gub
    g = d.constraint_leaf_g
    @constraint(
        model,
        leaf_poly[node=d.num_nonleaf_nodes+1:d.num_nodes],
        g_min .<= g * vcat(x[node_to_x(d, node)], u[node_to_u(d, node)]) .<= g_max
    )
end

function impose_state_input_constraints(
    model :: Model, 
    d :: Data, 
    )
    con = d.constraint_nonleaf
    if con == "rectangle" || con == "polyhedronWithIdentity"
        impose_rect_nonleaf(model, d)
    end
    if con == "polyhedron" || con == "polyhedronWithIdentity"
        impose_poly_nonleaf(model, d)
    end
    con = d.constraint_leaf
    if con == "rectangle" || con == "polyhedronWithIdentity"
        impose_rect_leaf(model, d)
    end
    if con == "polyhedron" || con == "polyhedronWithIdentity"
        impose_poly_leaf(model, d)
    end
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
            type_ = AVaR
        else
            throw("Risk type ($(d.risk_type)) not supported!")
        end

        if type_ == AVaR
            alpha = d.risk_alpha
            ch_num = d.ch_num[node]
            ch_probs = d.conditional_probabilities[node_to_ch(d, node)]  # Conditional probabilities of children
            eye = I(ch_num)  # Identity matrix
            dim = ch_num * 2 + 1
            E = vcat(alpha * eye, -eye, ones(1, ch_num))
            F = nothing  # Matrix F is not applicable for AVaR
            b = vcat(ch_probs, zeros(ch_num), 1)
            return new(type_, dim, E, F, b)
        else
            throw("Risk type not implemented!")
        end
    end
end

build_risk(d :: Data) = [Risk(d, node) for node in 1:d.num_nonleaf_nodes]

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
    # Add initial state constraint
    @constraint(
        model, 
        initial_condition[element = 1 : length(x0)], 
        x[element] .== x0[element]
    )
    # Solve problem
    optimize!(model)
end

export solve_this

end  # End of module
