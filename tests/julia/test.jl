import Pkg
Pkg.activate("tests/julia/modelFactory")
Pkg.instantiate()
include("modelFactory/src/modelFactory.jl")
using .modelFactory, JuMP, Gurobi, MosekTools, Ipopt, COSMO


data = read_data()
risk = build_risk(data)
x0 = read_vector_from_binary(TR, folder * "initialState" * file_ext_r)
tol::Float64 = 1e-3
max_time::Float64 = 5 * minute
status::TI = 1

time_g = 0.
time_m = 0.
time_i = 0.
time_c = 0.

model_g = build_model(Gurobi.Optimizer, data, risk)
set_attribute(model_g, "FeasibilityTol", tol)
set_attribute(model_g, "OptimalityTol", tol)
set_attribute(model_g, "TimeLimit", max_time)
println("(Gurobi) Solving ...")
try
    global time_g = @elapsed solve_this(model_g, x0)
catch e
    global time_g = 0.
    println(e)
end
if time_g > max_time
    time_g = 0.
end
status_g = termination_status(model_g)
println("(Gurobi) Done! ($(time_g) s) ($(status_g))")
if status_g == MOI.OPTIMAL || status_g == MOI.LOCALLY_SOLVED || status_g == MOI.TIME_LIMIT
    status = 0
end

if status == 0
    model_m = build_model(Mosek.Optimizer, data, risk)
    set_attribute(model_m, "MSK_DPAR_INTPNT_TOL_REL_GAP", tol)
    set_attribute(model_m, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", tol)
    set_attribute(model_m, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", tol)
    set_attribute(model_m, "MSK_DPAR_OPTIMIZER_MAX_TIME", max_time)
    println("(Mosek) Solving...")
    try
        global time_m = @elapsed solve_this(model_m, x0)
    catch e
        global time_m = 0.
        println(e)
    end
    if time_m > max_time
        time_m = 0.
    end
    println("(Mosek) Done! ($(time_m) s)")

    model_i = build_model(Ipopt.Optimizer, data, risk)
    set_attribute(model_i, "tol", tol)
    set_attribute(model_i, "max_cpu_time", max_time)
    set_attribute(model_i, "sb", "yes")
    println("(Ipopt) Solving...")
    try
        global time_i = @elapsed solve_this(model_i, x0)
    catch e
        global time_i = 0.
        println(e)
    end
    if time_i > max_time
        time_i = 0.
    end
    println("(Ipopt) Done! ($(time_i) s)")

    model_c = build_model(COSMO.Optimizer, data, risk)
    set_attribute(model_c, "eps_abs", tol)
    set_attribute(model_c, "eps_rel", tol)
    set_attribute(model_c, "time_limit", max_time)
    println("(Cosmo) Solving...")
    try
        global time_c = @elapsed solve_this(model_c, x0)
    catch e
        global time_c = 0.
        println(e)
    end
    if time_c > max_time
        time_c = 0.
    end
    println("(Cosmo) Done! ($(time_c) s)")

    println("Saving julia times ...")
    open("misc/timeCvxpy.csv", "a") do f
        write(f, "$(data.num_nodes), $(time_g), $(time_m), $(time_i), $(time_c), ")
    end
    println("Saved!")
end

exit(status)
