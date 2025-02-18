module help

"""
Variables
"""
U = UInt64  # unsigned ints
T = Float64  # double precision
folder = "data/"
file_type = ".bt"
file_ext_u = "_u" * file_type
if T == Float32
    file_ext_t = "_f" * file_type
else
    file_ext_t = "_d" * file_type
end

export U, T, folder, file_ext_u, file_ext_t

"""
    read_3D_array_from_binary(T, filename) -> Array{T,3}

Read a binary file where:
1) The first three 64-bit unsigned integers (UInt64) represent the dimensions (m, n, k).
2) The remaining data are the array elements of type `T` (in column-major format).
Returns a tensor of size (m, n, k) with element type `T`.
"""
function read_tensor_from_binary(::Type{T}, filename) where {T}
    open(filename, "r") do io
        # Read the first three UInt64 values for dimensions
        dims = Array{UInt64}(undef, 3)
        read!(io, dims)
        # Compute total number of elements
        num_el = prod(dims)
        # Read the remaining binary data as a Vector{T}
        data = Array{T}(undef, num_el)
        read!(io, data)
        # Reshape into a tensor
        return reshape(data, Int.(dims)...)
    end
end

export read_tensor_from_binary

end  # End of module
