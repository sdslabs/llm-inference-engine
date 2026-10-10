#if defined(USE_ROCM)
    #include <hip/hip_runtime.h>
    #include <hip/hipblas.h>

    #define cudaMalloc hipMalloc
    #define cudaFree hipFree
    #define cudaMemcpy hipMemcpy
    #define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
    #define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
    #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
    #define cudaMemset               hipMemset
    #define cudaStream_t             hipStream_t
    #define cudaStreamCreate         hipStreamCreate
    #define cudaStreamDestroy        hipStreamDestroy
    #define cudaStreamSynchronize    hipStreamSynchronize
    #define cudaDeviceSynchronize    hipDeviceSynchronize
    #define cudaError_t              hipError_t
    #define cudaSuccess              hipSuccess
    #define cudaGetErrorString       hipGetErrorString
    #define cudaGetLastError         hipGetLastError

    #define cublasHandle_t           hipblasHandle_t
    #define cublasCreate             hipblasCreate
    #define cublasDestroy            hipblasDestroy
    #define cublasSgemm              hipblasSgemm

    using bf16 = __hip_bfloat16;
#else
    #include <cuda_runtime.h>
    #include <cuda_bf16.h>
    #include <cublas_v2.h>

    using bf16 = __nv_bfloat16;
#endif