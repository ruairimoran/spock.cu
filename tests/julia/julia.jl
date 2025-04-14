import Pkg
Pkg.activate(joinpath(@__DIR__, "modelFactory"))
include(joinpath(@__DIR__, "modelFactory", "src", "modelFactory.jl"))
using .modelFactory, JuMP, Gurobi, MosekTools, Ipopt


function check_status(model :: Model, time, max_time)
    status = termination_status(model)
    if time > max_time || (status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED)
        time = 0.
    end
    return status, time
end


function run_model(model :: Model, max_time)
    println("[$(MOI.get(model, MOI.SolverName()))] Solving ...")
    time = 0.
    try
        time = @elapsed solve_this(model, x0)
    catch e
        println(e)
    end
    status, time = check_status(model, time, max_time)
    println("[$(MOI.get(model, MOI.SolverName()))] Done! ($(time) s) ($(status))")
    return status, time
end


"""
For example::random only!
Checks if any of the bounds are active.
"""
function check_bounds(model :: Model, status, d, tol)
    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
#         n = d.num_states + 1
#         sat_x_max = any(isapprox.(maximum(value.(model[:x])),
#             d.constraint_nonleaf_max[1]; atol = tol))
#         sat_x_min = any(isapprox.(minimum(value.(model[:x])),
#             d.constraint_nonleaf_min[1]; atol = tol))
#         sat_u_max = any(isapprox.(maximum(value.(model[:u])),
#             d.constraint_nonleaf_max[n]; atol = tol))
#         sat_u_min = any(isapprox.(minimum(value.(model[:u])),
#             d.constraint_nonleaf_min[n]; atol = tol))
#         sat_x = sat_x_max || sat_x_min
#         sat_u = sat_u_max || sat_u_min
#         println("[$(MOI.get(model, MOI.SolverName()))] Constraint saturation: states = ", sat_x, ", inputs = ", sat_u)
        if true  # print states and inputs
            println("[$(MOI.get(model, MOI.SolverName()))] States:")
            for node in range(1, d.num_nodes)
                println(value.(model[:x])[node*d.num_states-1:node*d.num_states])
            end
            println("[$(MOI.get(model, MOI.SolverName()))] Inputs:")
            for node in range(1, d.num_nonleaf_nodes)
                println(value.(model[:u])[node*d.num_inputs])
            end
        end
    end
end


function build_and_run(optimizer, data, risk, tol, max_time)
    model = build_model(optimizer, data, risk)
    if optimizer == Gurobi.Optimizer
        set_attribute(model, "FeasibilityTol", tol)
        set_attribute(model, "OptimalityTol", tol)
        set_attribute(model, "TimeLimit", max_time)
    elseif optimizer == Mosek.Optimizer
        set_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", tol_f64)
        set_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", tol_f64)
        set_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", tol_f64)
        set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", max_time_f64)
    elseif optimizer == Ipopt.Optimizer
        set_attribute(model, "tol", tol_f64)
        set_attribute(model, "max_cpu_time", max_time_f64)
        set_attribute(model, "sb", "yes")
    else
        println("Optimizer not set up!")
    end
    status, time = run_model(model, max_time)
    check_bounds(model, status, data, tol)
    return status, time
end


function get_stats(optimizer, data, risk, tol, max_time)
    status = Nothing
    time = 0.
    ram = 0.
    try
        (status, time), _, bytes, _, _ = @timed build_and_run(optimizer, data, risk, tol, max_time)
        ram = bytes / 1024^2
    catch e
        println(e)
    end
    return status, time, ram
end


function check_run(run, status)
    return run == 0 || status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
end


data = read_data()
risk = build_risk(data)
x0 = read_vector_from_binary(TR, folder * "initialState" * file_ext_r)
tol = 1e-3
max_time = .25 * minute
status = Int64(1)
tol_f64 = Float64(tol)
max_time_f64 = Float64(max_time)

status_m = nothing
status_g = nothing
status_i = nothing

for run in [0]
    if check_run(run, status_m)
        global status_m, time_m, ram_m = get_stats(Mosek.Optimizer, data, risk, tol_f64, max_time_f64)
        if status_m == MOI.INFEASIBLE
            break
        else
            global status = 0
        end
    end

#     if check_run(run, status_g)
#         global status_g, time_g, ram_g = get_stats(Gurobi.Optimizer, data, risk, tol_f64, max_time_f64)
#     end
#
#     if check_run(run, status_i)
#         global status_i, time_i, ram_i = get_stats(Ipopt.Optimizer, data, risk, tol_f64, max_time_f64)
#     end

    if run == 1
        println("Saving julia times...")
        num_vars = data.num_nodes * data.num_states + data.num_nonleaf_nodes * data.num_inputs
        open("time.csv", "a") do f
            # write(f, "$(data.ch_num[1]), $(time_g), $(time_m), $(time_i), ")
            # write(f, "$(data.ch_num[1]), $(time_g), $(ram_g), $(time_m), $(ram_m), $(time_i), $(ram_i), ")
            # write(f, "$(data.ch_num[1]), $(time_m), $(ram_m), ")
        end
        println("Saved!")
    end
end

exit(status)
