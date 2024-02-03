#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

template <typename T, size_t Extent = std::dynamic_extent>
	requires requires {
		// T must be trivially copyable as it will be bytewise copied to the device.
		std::is_trivially_copyable_v<T>;
	}
class CudaUniqueArray {

	public:
	static CudaUniqueArray alloc() requires (Extent != std::dynamic_extent) {
		T* ptr;
		cudaError_t result = cudaMalloc(&ptr, Extent * sizeof(T));
		if (result != cudaSuccess) {
			throw std::bad_alloc();
		}
		return CudaUniqueArray(std::span(ptr));
	}

	static CudaUniqueArray alloc(size_t size) requires (Extent == std::dynamic_extent) {
		T* ptr;
		cudaError_t result = cudaMalloc(&ptr, size * sizeof(T));
		if (result != cudaSuccess) {
			throw std::bad_alloc();
		}
		return CudaUniqueArray(std::span(ptr, size));
	}

	CudaUniqueArray(const CudaUniqueArray&) = delete;
	CudaUniqueArray(CudaUniqueArray&& other) noexcept
		: span(std::exchange(other.span, {})) {}

	CudaUniqueArray& operator=(const CudaUniqueArray&) = delete;
	CudaUniqueArray& operator=(CudaUniqueArray&& other) noexcept {
		std::swap(this->span, other.span);
		return *this;
	}

	~CudaUniqueArray() noexcept {
		cudaFree(this->span.data());
	}

	[[nodiscard]]
	__device__ std::span<T, Extent> get() const noexcept {
		return this->span;
	}

	[[nodiscard]]
	__device__ T* data() const noexcept {
		return this->span.data();
	}

	[[nodiscard]]
	size_t size() const noexcept {
		return this->span.size();
	}

	[[nodiscard]]
	size_t size_bytes() const noexcept {
		return this->span.size_bytes();
	}

	void upload(std::span<const T> src) const {
		cudaError_t result = cudaMemcpy(this->span.data(), src.data(), src.size_bytes(), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}
	}

	void download(std::span<T> dst) const {
		cudaError_t result = cudaMemcpy(dst.data(), this->span.data(), dst.size_bytes(), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(result));
		}
		return value;
	}

	private:
	CudaUniqueArray(std::span<T, Extent> span) : span(span) {}
	std::span<T, Extent> span;
};

#endif
