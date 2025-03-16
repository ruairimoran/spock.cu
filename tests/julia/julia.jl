import Pkg
Pkg.activate(joinpath(@__DIR__, "modelFactory"))
include(joinpath(@__DIR__, "modelFactory", "src", "modelFactory.jl"))
using .modelFactory, JuMP, Gurobi, MosekTools, Ipopt


function check_status(model :: Model, time, max_time)
    status = termination_status(model)
    if time > max_time || (status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED)
        time = 0.
    end
    println("[$(MOI.get(model, MOI.SolverName()))] Done! ($(time) s) ($(status))")
    return status, time
end


function run_model(model :: Model, max_t)
    println("[$(MOI.get(model, MOI.SolverName()))] Solving ...")
    t = 0.
    try
        t = @elapsed solve_this(model, x0)
    catch e
        println(e)
    end
    status, time = check_status(model, t, max_t)
    return status, time
end


function check_bounds(model :: Model, status, d, tol)
    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        n = d.num_states + 1
        sat_x_max = any(isapprox.(maximum(value.(model[:x])),
            d.constraint_nonleaf_max[1]; atol = tol))
        sat_x_min = any(isapprox.(minimum(value.(model[:x])),
            d.constraint_nonleaf_min[1]; atol = tol))
        sat_u_max = any(isapprox.(maximum(value.(model[:u])),
            d.constraint_nonleaf_max[n]; atol = tol))
        sat_u_min = any(isapprox.(minimum(value.(model[:u])),
            d.constraint_nonleaf_min[n]; atol = tol))
        sat_x = sat_x_max || sat_x_min
        sat_u = sat_u_max || sat_u_min
        println("[$(MOI.get(model, MOI.SolverName()))] Constraint saturation: states = ", sat_x, ", inputs = ", sat_u)
        if false  # print states and inputs
            println("[$(MOI.get(model, MOI.SolverName()))] States:")
            for state in value.(model[:x])
                println(state)
            end
            println("[$(MOI.get(model, MOI.SolverName()))] Inputs:")
            for input in value.(model[:u])
                println(input)
            end
        end
    end
end


data = read_data()
risk = build_risk(data)
x0 = read_vector_from_binary(TR, folder * "initialState" * file_ext_r)
tol = 1e-3
max_time = 5 * minute
status = Int64(1)

for run in [0, 1]
    time_g = 0.
    time_m = 0.
    time_i = 0.
    tol_f64 = Float64(tol)
    max_time_f64 = Float64(5 * minute)

    model_g = build_model(Gurobi.Optimizer, data, risk)
    set_attribute(model_g, "FeasibilityTol", tol_f64)
    set_attribute(model_g, "OptimalityTol", tol_f64)
    set_attribute(model_g, "TimeLimit", max_time_f64)
    status_g, time_g = run_model(model_g, max_time_f64)
    check_bounds(model_g, status_g, data, tol_f64)
    if status_g == MOI.OPTIMAL || status_g == MOI.LOCALLY_SOLVED ||
       status_g == MOI.TIME_LIMIT || status_g == MOI.NUMERICAL_ERROR
       global status = 0
    else
       break
    end

    model_m = build_model(Mosek.Optimizer, data, risk)
    set_attribute(model_m, "MSK_DPAR_INTPNT_TOL_REL_GAP", tol_f64)
    set_attribute(model_m, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", tol_f64)
    set_attribute(model_m, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", tol_f64)
    set_attribute(model_m, "MSK_DPAR_OPTIMIZER_MAX_TIME", max_time_f64)
    status_m, time_m = run_model(model_m, max_time_f64)
    check_bounds(model_m, status_m, data, tol_f64)

    model_i = build_model(Ipopt.Optimizer, data, risk)
    set_attribute(model_i, "tol", tol_f64)
    set_attribute(model_i, "max_cpu_time", max_time_f64)
    set_attribute(model_i, "sb", "yes")
    status_i, time_i = run_model(model_i, max_time_f64)
    check_bounds(model_i, status_i, data, tol_f64)

    if run == 1
        println("Saving julia times...")
        num_vars = data.num_nodes * (data.num_states + data.num_inputs)
        open("time.csv", "a") do f
            write(f, "$(num_vars), $(time_g), $(time_m), $(time_i), ")
        end
        println("Saved!")
    end
end

exit(status)
