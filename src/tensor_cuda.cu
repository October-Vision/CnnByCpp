#include "tensor.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cassert>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "\n[CUDA 致命错误] " << cudaGetErrorString(err) \
                      << " (位于文件 " << __FILE__ << " 第 " << __LINE__ << " 行)\n"; \
            exit(1); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "\n[cuBLAS 致命错误] 错误代码: " << status \
                      << " (位于文件 " << __FILE__ << " 第 " << __LINE__ << " 行)\n"; \
            exit(1); \
        } \
    } while (0)

// GPU 矩阵乘法: C = A * B
Tensor Tensor::dot_cuda(const Tensor& B) const {
    assert(shape.size() == 2 && B.shape.size() == 2 && "dot_cuda仅支持2D张量");
    assert(shape[1] == B.shape[0] && "矩阵维度不匹配");
    cudaFree(0);
    int M = shape[0];
    int N = shape[1];
    int P = B.shape[1];

    Tensor C({M, P}, 0.0f);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    //GPU显存中开辟空间
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, N * P * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, M * P * sizeof(float)));

    //拷贝数据到 GPU
    CHECK_CUDA(cudaMemcpy(d_A, this->data.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data.data(), N * P * sizeof(float), cudaMemcpyHostToDevice));

    // 执行 GPU 运算
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                P, M, N,          
                &alpha,
                d_B, P,           // 先传 B
                d_A, N,           // 再传 A
                &beta,
                d_C, P));

    //结果拷回
    CHECK_CUDA(cudaMemcpy(C.data.data(), d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost));

    //释放资源
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));

    return C;
}