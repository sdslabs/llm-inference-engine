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
    #define cudaGetDeviceProperties  hipGetDeviceProperties
    #define cudaDeviceProp           hipDeviceProp
    #define cudaMemGetInfo           hipGetMemInfo
    #define cudaDeviceGetAttribute   hipDeviceGetAttribute
    #define cudaGetErrorString       hipGetErrorString
    #define cudaGetLastError         hipGetLastError

    #define cublasHandle_t           hipblasHandle_t
    #define cublasCreate             hipblasCreate
    #define cublasDestroy            hipblasDestroy
    #define cublasSgemm              hipblasSgemm
    #define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
    #define CUDA_R_16BF              HIP_R_16BF
    #define CUBLAS_OP_T              HIPBLAS_OP_T
    #define CUBLAS_OP_N              HIPBLAS_OP_N
    #define CUBLAS_COMPUTE_32F       HIPBLAS_COMPUTE_32F
    #define CUBLAS_GEMM_DEFAULT      HIPBLAS_GEMM_DEFAULT

    using bf16 = __hip_bfloat16;
#else
    #include <cuda_runtime.h>
    #include <cuda_bf16.h>
    #include <cublas_v2.h>

    using bf16 = __nv_bfloat16;
#endif