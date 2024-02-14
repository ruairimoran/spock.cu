#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

template <typename T>
	requires requires {
		std::is_trivially_copyable_v<T>;
	}
class DeviceSlice;

/// Manages an array of objects allocated on the device.
/// The size of the array can be either a compile-time constant or a vaule specified at runtime.
/// When the `DeviceArray` is destroyed, the associated memory will automatically be freed.
template <typename T, size_t Extent = std::dynamic_extent>
	requires requires {
		// T must be trivially copyable as it will be bytewise copied to the device.
		std::is_trivially_copyable_v<T>;
		Extent != 0;
	}
class DeviceArray {

	public:

	/// Create a new `DeviceArray` with memory allocated on the device.
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

	/// Create a new `DeviceArray` with memory allocated on the device.
	/// This version allocates an amount of memory which is specified at runtime.
	/// @param size The size of the allocated array.
	/// @return A `DeviceArray` representing an uninitialised array of T in device memory.
	/// @exception std::bad_alloc The allocation was unsuccessful.
	static DeviceArray alloc(size_t size) requires (Extent == std::dynamic_extent) {
		if (size == 0) {
			return DeviceArray();
		}

		T* ptr = nullptr;
		cudaError_t result = cudaMalloc(&ptr, size * sizeof(T));
		if (result != cudaSuccess) {
			throw std::bad_alloc();
		}
		return DeviceArray(std::span<T, Extent>(ptr, size));
	}

	/// Create a new `DeviceArray` representing an empty array. This is only possible for arrays
	/// with dynamic extent. No memory will be allocated for this array.
	DeviceArray() noexcept requires (Extent == std::dynamic_extent) = default;

	DeviceArray(const DeviceArray&) = delete;
	DeviceArray(DeviceArray&& other) noexcept
		: span(std::move(other.span)) {}

	DeviceArray& operator=(const DeviceArray&) = delete;
	DeviceArray& operator=(DeviceArray&& other) noexcept {
		std::swap(this->span, other.span);
		return *this;
	}

	~DeviceArray() noexcept {
		cudaFree(this->span.data());
	}

	/// Create a new `DeviceArray` instance with its contents copied from this one.
	/// @return A `DeviceArray` which is a copy of this one.
	/// @exception std::bad_alloc The allocation was unsuccessful.
	/// @exception std::runtime_error An error occured during the copy.
	DeviceArray clone() const {
		DeviceArray cloned = [&] () {
			if constexpr (Extent == std::dynamic_extent) {
				return DeviceArray::alloc(this->size());
			} else {
				return DeviceArray::alloc();
			}
		}();

		cudaError_t result = cudaMemcpy(cloned.data(), this->data(), this->size_bytes(), cudaMemcpyDeviceToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}

		return cloned;
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
	/// @exception std::runtime_error An error occured during the copy.
	void upload(std::span<const T> src) const {
		cudaError_t result = cudaMemcpy(this->span.data(), src.data(), src.size_bytes(), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}
	}

	/// Copy data from device memory to host memory.
	/// @param dst The region of host memory to copy to.
	/// @exception std::runtime_error An error occured during the copy.
	void download(std::span<T> dst) const {
		cudaError_t result = cudaMemcpy(dst.data(), this->span.data(), dst.size_bytes(), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}
	}

	/// Create a non-owning view, or *slice*, of a subsequence of elements in this array.
	/// @param start The index of the first element in this slice.
	/// @param size  The number of elements in this slice. If not specified, it will default to
	///              the number of elements in the array after `start`.
	/// @return A `DeviceSlice` referring to a subsequence of elements in this array.
	/// @warning Since a slice does not have ownership of its contents, it must not be allowed to
	///          outlive this `DeviceArray`.
	inline DeviceSlice<T> slice(size_t start, size_t size = std::dynamic_extent) const noexcept;

	private:
	DeviceArray(std::span<T, Extent> span) : span(span) {}
	std::span<T, Extent> span;
};

/// `DeviceSlice` is a non-owning view of a sequence of objects in device memory. This sequence will
/// typically be created from a `DeviceArray` or a sub-slice of another `DeviceSlice`. It can also
/// be implicitly constructed from a `DeviceArray`.
/// @warning Since this class does not have ownership of its contents, it must not be allowed to
///          outlive the owner of those contents (e.g. the `DeviceArray` it was created from).
template <typename T>
	requires requires {
		std::is_trivially_copyable_v<T>;
	}
class DeviceSlice {

	public:

	/// Create a new `DeviceArray` instance with its contents copied from this slice.
	/// @return A `DeviceArray` which is an owned copy of this slice.
	/// @throw std::bad_alloc The allocation was unsuccessful.
	/// @throw std::runtime_error An error occured during the copy.
	DeviceArray<T> clone() const {
		DeviceArray cloned = DeviceArray<T>::alloc(this->size());

		cudaError_t result = cudaMemcpy(cloned.data(), this->data(), this->size_bytes(), cudaMemcpyDeviceToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}

		return cloned;
	}

	/// @return A span representing the slice in device memory.
	[[nodiscard]]
	std::span<T> get() const noexcept {
		return this->span;
	}

	/// @return A device pointer to the start of the slice, or nullptr if the slice is empty.
	[[nodiscard]]
	T* data() const noexcept {
		return this->span.data();
	}

	/// @return The number of elements in the slice.
	[[nodiscard]]
	size_t size() const noexcept {
		return this->span.size();
	}

	/// @return The size of the slice in bytes.
	[[nodiscard]]
	size_t size_bytes() const noexcept {
		return this->span.size_bytes();
	}

	/// @return `true` if the slice is empty, `false` otherwise.
	[[nodiscard]]
	bool empty() const noexcept {
		return this->span.empty();
	}

	/// Copy data from host memory to device memory.
	/// @param src The region of host memory to copy from.
	/// @exception std::runtime_error An error occured during the copy.
	void upload(std::span<const T> src) const {
		cudaError_t result = cudaMemcpy(this->span.data(), src.data(), src.size_bytes(), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}
	}

	/// Copy data from device memory to host memory.
	/// @param dst The region of host memory to copy to.
	/// @exception std::runtime_error An error occured during the copy.
	void download(std::span<T> dst) const {
		cudaError_t result = cudaMemcpy(dst.data(), this->span.data(), dst.size_bytes(), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}
	}

	/// Create a sub-slice of the elements in this slice.
	/// @param start The index of the first element in this slice.
	/// @param size  The number of elements in this slice. If not specified, it will default to
	///              the number of elements in the slice after `start`.
	/// @return A `DeviceSlice` referring to a subsequence of elements in this slice.
	DeviceSlice<T> slice(size_t start, size_t size = std::dynamic_extent) const noexcept {
		return DeviceSlice<T>(this->span.subspan(start, size));
	}

	DeviceSlice() = default;
	explicit DeviceSlice(std::span<T> span) : span(span) {}
	DeviceSlice(const DeviceArray<T>& array) : span(array.get()) {}

	private:
	std::span<T> span;

};


template <typename T, size_t Extent>
	requires requires {
		std::is_trivially_copyable_v<T>;
		Extent != 0;
	}
DeviceSlice<T> DeviceArray<T, Extent>::slice(size_t start, size_t size) const noexcept {
	return DeviceSlice<T>(this->span.subspan(start, size));
}

#endif
