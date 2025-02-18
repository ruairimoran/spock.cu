include("help.jl")
using .help


arrU = read_tensor_from_binary(U, folder * "ancestors" * file_ext_u)
arrT = read_tensor_from_binary(T, folder * "dynamics_AB" * file_ext_t)


