import Pkg
Pkg.activate("tests/julia/modelFactory")
Pkg.instantiate()
include("modelFactory/src/modelFactory.jl")
using .modelFactory  # , MosekTools, Gurobi, Ipopt, SeDuMi, COSMO

arrU = read_tensor_from_binary(U, folder * "ancestors" * file_ext_u)
arrT = read_tensor_from_binary(T, folder * "dynamics_AB" * file_ext_t)
build_model()


# N_min = 5
# N_max = 15
# s = N_max - N_min + 1
# # Dimensions of state and input vectors
# nx = 10
# x0 = [.9 for i = 1:nx]
# d = 3
# TOL = 1e-3
#
# scen_tree, cost, dynamics, rms, constraints = get_server_heat_specs(N, nx, d)
#
# num_nodes[idx] = scen_tree.n
#
# model = spock.build_model(scen_tree, cost, dynamics, rms, constraints, spock.SolverOptions(spock.SP, nothing))
#
# model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms, constraints)
# set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_TOL_REL_GAP", TOL)
# set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", TOL)
# set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", TOL)
#
# model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms, constraints)
# set_optimizer_attribute(model_gurobi, "FeasibilityTol", TOL)
# set_optimizer_attribute(model_gurobi, "OptimalityTol", TOL)
#
# model_ipopt = spock.build_model_ipopt(scen_tree, cost, dynamics, rms, constraints)
# set_optimizer_attribute(model_ipopt, "tol", TOL)
#
# model_sedumi = spock.build_model_sedumi(scen_tree, cost, dynamics, rms, constraints)
#
# model_cosmo = spock.build_model_cosmo(scen_tree, cost, dynamics, rms, constraints)
# set_optimizer_attribute(model_cosmo, "eps_abs", TOL)
# set_optimizer_attribute(model_cosmo, "eps_rel", TOL)
#
# ##########################################
# ###  Solution
# ##########################################
#
# println("solving...")
#
# if maximum(model_timings) <= t_max
#   model_timings[idx] += @elapsed spock.solve_model!(model, x0, tol=TOL)
# end
# if maximum(mosek_timings) <= t_max
#   mosek_timings[N - 2] += @elapsed spock.solve_model(model_mosek, x0)
# end
# if maximum(gurobi_timings) <= t_max
#   gurobi_timings[N - 2] += @elapsed spock.solve_model(model_gurobi, x0)
# end
# if maximum(ipopt_timings) <= t_max && N <= 12
#   ipopt_timings[N - 2] += @elapsed spock.solve_model(model_ipopt, x0)
# end
# if maximum(cosmo_timings) <= t_max
#   cosmo_timings[N - 2] += @elapsed spock.solve_model(model_cosmo, x0)
# end
# if maximum(sedumi_timings) <= t_max
#   sedumi_timings[N - 2] += @elapsed spock.solve_model(model_sedumi, x0)
# end
#
# mosek_timings = filter(>(0.), mosek_timings)
# gurobi_timings = filter(>(0.), gurobi_timings)
# ipopt_timings = filter(>(0.), ipopt_timings)
# sedumi_timings = filter(>(0.), sedumi_timings)
# cosmo_timings = filter(>(0.), cosmo_timings)
#
# open("misc/timeCvxpy.csv", "a") do f
#   write(f, "$(num_nodes), $(t_gurob), $(t_mosek), $(t_ipopt), $(t_sedum), $(t_cosmo)")
# end

status = 0
exit(status)