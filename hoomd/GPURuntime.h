// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_RUNTIME_H_
#define NEIGHBOR_RUNTIME_H_

#include <functional>
#include <utility>

#include <cuda_runtime.h>
#ifdef __CUDACC__
#define GPUCC
#endif

namespace neighbor
{
namespace gpu
{

#if defined (__GPU_PLATFORM_NVIDIA__)
/* errors */
typedef cudaError_t error_t;
enum error
    {
    success = cudaSuccess
    };

#elif defined(__GPU_PLATFORM_HIP__)
typedef hipError_t error_t;
enum error
    {
    success = hipSuccess
    };

#endif

//! Coerce comparison of gpu::error with the native error as int
inline bool operator==(error a, error_t b)
    {
    return (static_cast<int>(a) == static_cast<int>(b));
    }
//! Coerce comparison of gpu::error with the native error as int
inline bool operator==(error_t a, error b)
    {
    return (static_cast<int>(a) == static_cast<int>(b));
    }

/* streams */
#if defined(__GPU_PLATFORM_NVIDIA__)
typedef cudaStream_t stream_t;
#elif defined(__GPU_PLATFORM_HIP__)
typedef hipStream_t stream_t;
#endif

//! Create a GPU stream
inline error_t streamCreate(stream_t* stream)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaStreamCreate(stream);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipStreamCreate(stream);
    #endif
    }

//! Synchronize a GPU stream
inline error_t streamSynchronize(stream_t stream)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaStreamSynchronize(stream);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipStreamSynchronize(stream);
    #endif
    }

//! Destroy a GPU stream
inline error_t streamDestroy(stream_t stream)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaStreamDestroy(stream);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipStreamDestroy(stream);
    #endif
    }

/*events*/
#if defined(__GPU_PLATFORM_NVIDIA__)
typedef cudaEvent_t event_t;
#elif defined(__GPU_DEFINED_HIP__)
typedef hipEvent_t event_t;
#endif

//!Create a GPU event
inline error_t eventCreate(event_t* event)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaEventCreate(event);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipEventCreate(event);
    #endif
    }

//!Create a GPU event with flags
inline error_t eventCreateWithFlags(event_t* event, unsigned int flags)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaEventCreateWithFlags(event, flags);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipEventCreateWithFlages(event, flags);
    #endif
    }i

#if defined(__GPU_PLATFORM_NVIDIA__)
unsigned int eventDisableTiming = cudaEventDisableTiming;
#elif defined(__GPU_PLATFORM_HIP__)
unsigned int eventDisableTiming = hipEventDisableTiming;
#endif

//!Destroy a GPU event
inline error_t eventDestroy(event_t event)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaEventDestroy(event);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipEventDestroy(event);
    #endif
    }

//!Synchronize a GPU event
inline errot_t eventSynchronize(event_t event)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaEventSychronize(event);
    #elif defined(__GPU_PLATFORM_HIP__)
    return cudaEventSynchronize(event);
    #endif
    }

//!Records GPU event in a GPU stream
inline error_t eventRecord(event_t event, stream_t stream = 0)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaEventRecord(event, stream)
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipEventRecord(event, stream)
    #endif
    }

//!Get Elapsed Time between two GPU events
inline eventElaspedTime(float* ms, event_t start, event_t end)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaEventElapsedTime(ms, start, end);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipEventElapsedTime(ms, start, end);
    #endif
    }

inline streamWaitEvent(stream_t stream, event_t event, unsigned int flags)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaStreamWaitEvent(stream, event, flags);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipStreamWaitEvent(stream, event, flags);
    #endif
    }

/*Device*/
#if defined(__GPU_PLATFORM_NVIDIA__)
typedef cudaDeviceProp deviceProp_t;
#elif defined(__GPU_PLATFORM_HIP__)
typedef hipDeviceProp_t deviceProp_t;
#endif

//!Synchronize Device
inline error_t deviceSynchronize(void)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaDeviceSynchronize();
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipDeviceSynchronize();
    #endif
    }

//!Get device currently being used
inline error_t getDevice(int* device)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaGetDevice(device);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipGetDevice(device);
    #endif
    }

//!Get number of "compute-capable" devices
inline error_t getDeviceCount(int* count)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaGetDeviceCount(count);
    #elif defined(_GPU_PLATFORM_HIP__)
    return hipGetDeviceCount(count);
    #endif
    }

//!Set device to use
inline error_t setDevice(int device)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaSetDevice(device);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipSetDevice(device);
    #endif
    }

//!Get properties of device
inline error_t getDeviceProperties(deviceProp_t* prop, int device)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaGetDeviceProperties(prop, device);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipGetDeviceProperties(prop, device);
    #endif
    }

//!Set list vaild devices(CUDA only)
#if defined(__GPU_PLATFORM_NVIDIA__)
inline error_t setValidDevices(int* device_arr. int len)
    {
        cudaSetValidDevices(device_arr, len);
    }
#endif

/*Threads*/
//!Synchronize threads (CUDA only)
#if defined(__GPU_PLATFORM_NVIDIA__)
inline error_t threadSynchronize(void)
    {
    return cudaThreadSynchronize();
    }
#endif

/* memory */
#if defined(__GPU_PLATFORM_NVIDIA__)
static const int memAttachGlobal = cudaMemAttachGlobal;
static const int memAttachHost = cudaMemAttachHost;
#elif defined(__GPU_PLATFORM_HIP__)
static const int memAttachGlobal = hipMemAttachGlobal;
static const int memAttachHost = hipMemAttachHost;
#endif

//! Allocate GPU memory
inline error_t malloc(void** ptr, size_t size)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaMalloc(ptr, size);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipMalloc(ptr, size);
    #endif
    }

//! Allocate managed GPU memory
inline error_t mallocManaged(void** ptr, size_t size, unsigned int flags = memAttachGlobal)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaMallocManaged(ptr, size, flags);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipMallocManaged(ptr, size, flags);
    #endif
    }

//! hostMallocFlag
#if defined(__GPU_PLATFORM_HIP__)
#define hostMallocDefault hipHostMallocDefault
#endif
//! Allocate Host Memory (hip only)
#if defined(__GPU_PLATFORM_HIP__)
inline error_t hostMalloc(void** ptr, size_t size, unsigned int flags = hostMallocDefault)
    {
    return hipHostMalloc(ptr, size, flags);
    }
#endif

//! Free GPU memory
inline error_t free(void* ptr)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaFree(ptr);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipFree(ptr);
    #endif
    }

//! Free host memory
inline error_t hostFree(void* ptr)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaHostFree(ptr);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipHostFree(ptr);
    #endif
    }

//! Set GPU memory to a value
inline error_t memset(void* ptr, int value, size_t count)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaMemset(ptr, value, count);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipMemset(ptr, value, count);
    #endif
    }

//! Asynchronously set GPU memory to a value
inline error_t memsetAsync(void* ptr, int value, size_t count, stream_t stream=0)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaMemsetAsync(ptr, value, count, stream);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipMemsetAsync(ptr, value, count, stream);
    #endif
    }

//!Resgister Host data for GPU use
inline error_t hostRegister(void* ptr, size_t size, unsigned int flags)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaHostRegister(ptr, size, flags);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipHostRegister(ptr, size, flags);
    #endif
    }

//!Unregister Host data
inline error_t hostUnregister(void *ptr)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaHostUnregister(ptr);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipHostUnregister(ptr);
    #endif
    }

//!Pass back device pointer of mapped host memory, which is registered in hostRegister or allocated in hostMalloc
inline error_t hostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaHostGetDevicePointer(pDevice, pHost, flags);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipHostGetDevicePointer(pDevice, pHost, flags);
    #endif
    }

//!Memcpy Kind enum
#if defined(__GPU_PLATFORM_NVIDIA__)
typedef enum memcpyKind
    {
    memcpyHostToHost = cudaMemcpyHostToHost,
    memcpyHostToDevice = cudaMemcpyHostToDevice,
    memcpyDeviceToHost = cudaMemcpyDeviceToHost,
    memcpyDeviceToDevice = cudaMemcpyDeviceToDevice,
    memcpyDefault = cudaMemcpyDefault
    };
#elif defined(__GPU_PLATFROM_HIP__)
typedef enum memcpyKind
    {
    memcpyHostToHost = hipMemcpyHostToHost,
    memcpyHostToDevice = hipMemcpyHostToDevice,
    memcpyDeviceToHost = hipMemcpyDeviceToHost,
    memcpyDeviceToDevice = hipMemcpyDeviceToDevice,
    memcpyDefault = hipMemcpyDefault
    };
#endif

//Copy data between host and device
inline error_t memcpy(void* dst, const void* src, size_t count, memcpyKind kind)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaMemcpy(dst, src, count, kind);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipMemcpy(dst, src, count, kind);
    #endif
    }

//!Asynchronously copy memory between host and device
inline error_t memcpyAsync(void* dst, const void* src, size_t count, memcpyKind kind, stream_t stream = 0)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaMemcpyAsync(dst, src, count, kind, stream);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipMemcpyAsync(dst, src, count, kind, stream);
    #endif
    }

//!memAdvise enum (CUDA only)
#if defined(__GPU_PLATFORM_NVIDIA__)
typedef enum memoryAdvise
    {
    memAdviseSetReadMostly = cudaMemAdviseSetReadMostly,
    memAdviseUnsetReadMostly = cudaMemAdviseUnsetReadMostly,
    memAdviseSetPreferredLocation = cudaMemAdviseSetPreferredLocation,
    memAdviseUnsetPreferredLocation = cudamemAdviseUnsetPreferredLocation,
    memAdviseSetAccessedBy = cudaMemAdviseSetAccessedBy,
    memAdviseUnsetAccessedBy = cudaMemAdviseUnsetAccessedBy
    }
#endif

//!Advise memory management (CUDA only)
#if defined(__GPU_PLATFORM_NIVIDIA__)
inline memAdvise(const void* devPtr, size_t size, memoryAdvise advise, int device)
    {
    cudaMemAdvise(devPtr, size, advise, device);
    }
#endif

/*Error Handling*/
inline error_t getErrorString(error_t error)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaGetErrorString(error);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipGetErrorString(error);
    #endif
    }

inline error_t getLastError(void)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaGetLastError();
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipGetLastError();
    #endif
    }

/* kernels */
#if defined(__GPU_PLATFORM_NVIDIA__)
typedef cudaFuncAttributes funcAttributes;
#elif defined(__GPU_PLATFORM_HIP__)
typedef hipFuncAttributes funcAttributes;
#endif

//! Get the GPU function attributes
inline error_t funcGetAttributes(funcAttributes* attr, const void* func)
    {
    #if defined(__GPU_PLATFORM_NVIDIA__)
    return cudaFuncGetAttributes(attr, func);
    #elif defined(__GPU_PLATFORM_HIP__)
    return hipFuncGetAttributes(attr, func);
    #endif
    }

#ifdef GPUCC
//! Launch a compute kernel on the GPU
class KernelLauncher
    {
    public:
        KernelLauncher(int blocks, int threadsPerBlock, size_t sharedBytes, stream_t stream)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(sharedBytes),
              stream_(stream)
            {}

        KernelLauncher(int blocks, int threadsPerBlock, size_t sharedBytes)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(sharedBytes),
              stream_(0)
            {}

        KernelLauncher(int blocks, int threadsPerBlock, stream_t stream)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(0),
              stream_(stream)
            {}

        KernelLauncher(int blocks, int threadsPerBlock)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(0),
              stream_(0)
            {}

        KernelLauncher(dim3 blocks, dim3 threadsPerBlock, size_t sharedBytes, stream_t stream)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(sharedBytes),
              stream_(stream)
            {}

        KernelLauncher(dim3 blocks, dim3 threadsPerBlock, size_t sharedBytes)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(sharedBytes),
              stream_(0)
            {}

        KernelLauncher(dim3 blocks, dim3 threadsPerBlock, stream_t stream)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(0),
              stream_(stream)
            {}

        KernelLauncher(dim3 blocks, dim3 threadsPerBlock)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(0),
              stream_(0)
            {}

        template<class Kernel, class ...Args>
        void operator()(const Kernel& kernel, Args&&... args)
            {
            kernel<<<blocks_,threadsPerBlock_,sharedBytes_,stream_>>>(std::forward<Args>(args)...);
            }

    private:
        dim3 blocks_;
        dim3 threadsPerBlock_;
        size_t sharedBytes_;
        stream_t stream_;
    };
#endif
} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_RUNTIME_H_
