#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

/// Manages an array of objects allocated on the device.
/// The size of the array can be either a compile-time constant or a vaule specified at runtime.
/// When the DeviceArray is destroyed, the associated memory will automatically be freed.
template <typename T, size_t Extent = std::dynamic_extent>
	requires requires {
		// T must be trivially copyable as it will be bytewise copied to the device.
		std::is_trivially_copyable_v<T>;
	}
class DeviceArray {

	public:

	/// Create a new DeviceInstance with memory allocated on the device.
	/// This version allocates an amount of memory which is fixed at compile time.
	/// @return A DeviceArray representing an uninitialised array of T in device memory.
	/// @exception std::bad_alloc if the allocation was unsuccessful.
	static DeviceArray alloc() requires (Extent != std::dynamic_extent) {
		T* ptr = nullptr;
		cudaError_t result = cudaMalloc(&ptr, Extent * sizeof(T));
		if (result != cudaSuccess) {
			throw std::bad_alloc();
		}
		return DeviceArray(std::span<T, Extent>(ptr, Extent));
	}

	/// Create a new DeviceInstance with memory allocated on the device.
	/// This version allocates an amount of memory which is specified at runtime.
	/// @param size The size of the allocated array.
	/// @return A DeviceArray representing an uninitialised array of T in device memory.
	/// @exception std::bad_alloc The allocation was unsuccessful.
	static DeviceArray alloc(size_t size) requires (Extent == std::dynamic_extent) {
		T* ptr = nullptr;
		cudaError_t result = cudaMalloc(&ptr, size * sizeof(T));
		if (result != cudaSuccess) {
			throw std::bad_alloc();
		}
		return DeviceArray(std::span<T, Extent>(ptr, size));
	}

	/// Create a new DeviceInstance representing an empty array. This is only possible for arrays
	/// with dynamic extent. No memory will be allocated for this array.
	DeviceArray() noexcept requires (Extent == std::dynamic_extent) = default;

	DeviceArray(const DeviceArray&) = delete;
	DeviceArray(DeviceArray&& other) noexcept
		: span(std::exchange(other.span, {})) {}

	DeviceArray& operator=(const DeviceArray&) = delete;
	DeviceArray& operator=(DeviceArray&& other) noexcept {
		std::swap(this->span, other.span);
		return *this;
	}

	~DeviceArray() noexcept {
		cudaFree(this->span.data());
	}

	/// @return A span representing the allocated device memory.
	[[nodiscard]]
	std::span<T, Extent> get() const noexcept {
		return this->span;
	}

	/// @return A device pointer to the allocated memory, or nullptr if the array is empty.
	[[nodiscard]]
	T* data() const noexcept {
		return this->span.data();
	}

	/// @return The number of elements in the array.
	[[nodiscard]]
	size_t size() const noexcept {
		return this->span.size();
	}

	/// @return The size of the array in bytes.
	[[nodiscard]]
	size_t size_bytes() const noexcept {
		return this->span.size_bytes();
	}

	/// @return `true` if the array is empty, `false` otherwise.
	[[nodiscard]]
	bool empty() const noexcept {
		return this->span.empty();
	}

	/// Copy data from host memory to device memory.
	/// @param src The region of host memory to copy from.
	/// @exception std::runtime_error if an error occurs during the copy.
	void upload(std::span<const T> src) const {
		cudaError_t result = cudaMemcpy(this->span.data(), src.data(), src.size_bytes(), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}
	}

	/// Copy data from device memory to host memory.
	/// @param dst The region of host memory to copy to.
	/// @exception std::runtime_error if an error occurs during the copy.
	void download(std::span<T> dst) const {
		cudaError_t result = cudaMemcpy(dst.data(), this->span.data(), dst.size_bytes(), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}
	}

	private:
	DeviceArray(std::span<T, Extent> span) : span(span) {}
	std::span<T, Extent> span;
};

#endif
