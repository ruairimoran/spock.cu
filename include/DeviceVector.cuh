#ifndef __DEVICE_VECTOR__
#define __DEVICE_VECTOR__


/**
 * DeviceVector is a unique_ptr-type entity for device data.
 */
template<typename TElement>
class DeviceVector {

    private:
        /** Pointer to device data */
        TElement *m_d_data = nullptr;
        /** Number of allocated elements */
        size_t m_numAllocatedElements = 0;
        bool m_doDestroy = false;

        bool destroy() {
            if (!m_doDestroy) return false;
            if (m_d_data) cudaFree(m_d_data);
            m_numAllocatedElements = 0;
            m_d_data = nullptr;
            return true;
        }

    public:
        /**
         * Constructs a DeviceVector object
         */
        DeviceVector() = default;

        /**
         * Constructs a DeviceVector object and allocates
         * memory on the device for n elements
         */
        DeviceVector(size_t n) {
            allocateOnDevice(n);
        }

        /**
         * Take a slice of another DeviceVector
         *
         * @param other other device vector
         * @param from start (index)
         * @param to end
         */
        DeviceVector(DeviceVector &other, size_t from, size_t to) {
            m_doDestroy = false;
            m_numAllocatedElements = to - from + 1;
            m_d_data = other.m_d_data + from;
        }

        /**
         * Create device vector from host vector.
         * This allocates memory on the device and copies the host data.
         *
         * @param vec host vector
         */
        DeviceVector(const std::vector<TElement> &vec) {
            allocateOnDevice(vec.size());
            upload(vec);
        }

        /**
         * Destructor - frees the device memory
         */
        ~DeviceVector() {
            destroy();
        }

        /**
         * Allocates memory on the device for `size` elements
         *
         * @param size number of elements
         * @return true if and only if no errors occured during
         *         the memory allocation
         */
        bool allocateOnDevice(size_t size);

        /**
         * Size of allocated memory space on the device
         */
        size_t capacity() {
            return m_numAllocatedElements;
        }


        /**
         * Upload array of data to device
         *
         * Note that if the allocated memory is insufficient,
         * it will be attempted to allocate new memory on the
         * device after freeing the previously allocated memory.
         *
         * @param dataArray pointer to array of data
         * @param size size of array
         * @return true iff the uploading is successful
         */
        bool upload(const TElement *dataArray, size_t size);


        /**
         * Uploads a the data of vector to the device
         *
         * @param vec vector to be uploaded
         * @return true iff the uploading is successful
         */
        bool upload(const std::vector<TElement>& vec)
        {
            return upload(vec.data(), vec.size());
        }

        /**
         * Returns the raw pointer to the device data
         */
        TElement *get() {
            return m_d_data;
        }

        /**
         * Downloads the device data to a provided host
         * memory position. It is assumed the memory position
         * on the host is appropriately allocated for the
         * device data to be copied.
         *
         * @param hostData destination memory position on host
         */
        void download(TElement* hostData);

        /**
         * Download the device data to a vector
         *
         * @param vec
         */
        void download(std::vector<TElement> &vec);

        /**
         * Fetches just one value from the device
         *
         * Use sparingly
         *
         * @param i index
         * @return entry of array at index i
         */
        TElement fetchElementFromDevice(size_t i);

        /**
         * Copy data to another memory position on the device.
         *
         * @param elsewhere destination
         */
        void deviceCopyTo(DeviceVector<TElement> &elsewhere);

}; /* end of class */


template<typename TElement>
TElement DeviceVector<TElement>::fetchElementFromDevice(size_t i) {
    DeviceVector<TElement> d_element(*this, i, i);
    TElement xi[1];
    d_element.download(xi);
    return xi[0];
}

template<typename TElement>
bool DeviceVector<TElement>::allocateOnDevice(size_t size) {
    if (size <= 0) return false;
    if (size <= m_numAllocatedElements) return true;
    destroy();
    m_doDestroy = true;
    size_t buffer_size = size * sizeof(TElement);
    bool cudaStatus = cudaMalloc(&m_d_data, buffer_size);
    if (cudaStatus != cudaSuccess) return false;
    m_numAllocatedElements = size;
    return true;
}


template<typename TElement>
bool DeviceVector<TElement>::upload(const TElement *dataArray, size_t size) {
    if (!allocateOnDevice(size)) return false;
    if (size <= m_numAllocatedElements) {
        size_t buffer_size = size * sizeof(TElement);
        cudaMemcpy(m_d_data, dataArray, buffer_size, cudaMemcpyHostToDevice);
    }
    return true;
}

template<typename TElement>
void DeviceVector<TElement>::deviceCopyTo(DeviceVector<TElement>& elsewhere) {
    elsewhere.allocateOnDevice(m_numAllocatedElements);
    cudaMemcpy(elsewhere.get(),
               m_d_data,
               m_numAllocatedElements * sizeof(TElement),
               cudaMemcpyDeviceToDevice);
}

template<typename TElement>
void DeviceVector<TElement>::download(TElement* hostData) {
    cudaMemcpy(hostData,
               m_d_data,
               m_numAllocatedElements * sizeof(TElement),
               cudaMemcpyDeviceToHost);
}

template<typename TElement>
void DeviceVector<TElement>::download(std::vector<TElement>& vec) {
    vec.resize(m_numAllocatedElements);
    cudaMemcpy(vec.data(),
               m_d_data,
               m_numAllocatedElements * sizeof(TElement),
               cudaMemcpyDeviceToHost);
}


#endif
