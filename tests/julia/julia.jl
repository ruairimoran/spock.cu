import Pkg
Pkg.activate(joinpath(@__DIR__, "modelFactory"))
include(joinpath(@__DIR__, "modelFactory", "src", "modelFactory.jl"))
using .modelFactory, JuMP, Gurobi, MosekTools, Ipopt  # COSMO


function check_status(model :: Model, time, max_time)
    status = termination_status(model)
    if time > max_time || (status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED)
        time = 0.
    end
    println("Done! ($(time) s) ($(status))")
    return status, time
end


data = read_data()
risk = build_risk(data)
x0 = read_vector_from_binary(TR, folder * "initialState" * file_ext_r)
tol = 1e-3
max_time = 5 * minute


time_g = 0.
time_m = 0.
time_i = 0.
# time_c = 0.
tol_f64 = Float64(1e-3)
max_time_f64 = Float64(5 * minute)
status = Int64(1)

model_g = build_model(Gurobi.Optimizer, data, risk)
set_attribute(model_g, "FeasibilityTol", tol_f64)
set_attribute(model_g, "OptimalityTol", tol_f64)
set_attribute(model_g, "TimeLimit", max_time_f64)
println("(Gurobi) Solving ...")
try
    global time_g = @elapsed solve_this(model_g, x0)
catch e
    global time_g = 0.
    println(e)
end
status_g, time_g = check_status(model_g, time_g, max_time_f64)
if status_g == MOI.OPTIMAL || status_g == MOI.LOCALLY_SOLVED || status_g == MOI.TIME_LIMIT
    status = 0
end

if status == 0
    atol = 1e-3
    sat_x_max = any(isapprox.(maximum(value.(model_g[:x])), data.constraint_nonleaf_max[1]; atol=atol))
    sat_x_min = any(isapprox.(minimum(value.(model_g[:x])), data.constraint_nonleaf_min[1]; atol=atol))
    sat_u_max = any(isapprox.(maximum(value.(model_g[:u])), data.constraint_nonleaf_max[data.num_states + 1]; atol=atol))
    sat_u_min = any(isapprox.(minimum(value.(model_g[:u])), data.constraint_nonleaf_min[data.num_states + 1]; atol=atol))
    sat_x = sat_x_max || sat_x_min
    sat_u = sat_u_max || sat_u_min
    println("(Gurobi) Constraint saturation: states = ", sat_x, ", inputs = ", sat_u)

    model_m = build_model(Mosek.Optimizer, data, risk)
    set_attribute(model_m, "MSK_DPAR_INTPNT_TOL_REL_GAP", tol_f64)
    set_attribute(model_m, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", tol_f64)
    set_attribute(model_m, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", tol_f64)
    set_attribute(model_m, "MSK_DPAR_OPTIMIZER_MAX_TIME", max_time_f64)
    println("(Mosek) Solving...")
    try
        global time_m = @elapsed solve_this(model_m, x0)
    catch e
        global time_m = 0.
        println(e)
    end
    status_m, time_m = check_status(model_m, time_m, max_time_f64)

    model_i = build_model(Ipopt.Optimizer, data, risk)
    set_attribute(model_i, "tol", tol_f64)
    set_attribute(model_i, "max_cpu_time", max_time_f64)
    set_attribute(model_i, "sb", "yes")
    println("(Ipopt) Solving...")
    try
        global time_i = @elapsed solve_this(model_i, x0)
    catch e
        global time_i = 0.
        println(e)
    end
    status_i, time_i = check_status(model_i, time_i, max_time_f64)

#     model_c = build_model(COSMO.Optimizer, data, risk)
#     set_attribute(model_c, "eps_abs", tol_f64)
#     set_attribute(model_c, "eps_rel", tol_f64)
#     set_attribute(model_c, "time_limit", max_time_f64)
#     println("(Cosmo) Solving...")
#     try
#         global time_c = @elapsed solve_this(model_c, x0)
#     catch e
#         global time_c = 0.
#         println(e)
#     end
#     status_c, time_c = check_status(model_c, time_c, max_time_f64)

    println("Saving julia times ...")
    num_vars = data.num_nodes * (data.num_states + data.num_inputs)
    open("time.csv", "a") do f
        write(f, "$(num_vars), $(time_g), $(time_m), $(time_i), ")
#         write(f, "$(num_vars), $(time_g), $(time_m), $(time_i), $(time_c), ")
    end
    println("Saved!")
end

exit(status)
