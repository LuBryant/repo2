#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 定义矩阵尺寸
#define MATRIX_SIZE 1024

// CUDA核函数：执行矩阵乘法
__global__ void matrixMultiply(float *A, float *B, float *C, int size) {
    // 计算当前线程对应的矩阵元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保线程不超出矩阵边界
    if (row < size && col < size) {
        float sum = 0.0f;
        // 计算矩阵乘法的一个元素
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

// 初始化矩阵
void initMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// 打印矩阵（仅打印左上角的一小部分以验证结果）
void printMatrix(float *matrix, int size, const char* name) {
    printf("%s matrix (showing top-left 5x5 corner):\n", name);
    int display_size = size < 5 ? size : 5;
    for (int i = 0; i < display_size; i++) {
        for (int j = 0; j < display_size; j++) {
            printf("%.2f\t", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// CPU版本的矩阵乘法，用于验证结果
void cpuMatrixMultiply(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main() {
    // 设置矩阵大小
    int size = MATRIX_SIZE;
    size_t bytes = size * size * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    
    // 初始化输入矩阵
    initMatrix(h_A, size);
    initMatrix(h_B, size);
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // 将输入矩阵从主机内存复制到设备内存
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // 定义线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // 创建CUDA事件来计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    
    // 启动核函数
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    
    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算经过的时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // 计算CPU版本的矩阵乘法（仅用于小矩阵时验证结果）
    if (size <= 1024) {
        printf("Computing CPU version for verification...\n");
        cpuMatrixMultiply(h_A, h_B, h_C_cpu, size);
        
        // 验证结果
        bool correct = true;
        for (int i = 0; i < size * size; i++) {
            if (fabs(h_C[i] - h_C_cpu[i]) > 1e-5) {
                correct = false;
                break;
            }
        }
        printf("GPU and CPU results %s\n", correct ? "match" : "don't match");
    }
    
    // 打印部分结果
    if (size <= 1024) {
        printMatrix(h_A, size, "Input A");
        printMatrix(h_B, size, "Input B");
        printMatrix(h_C, size, "Output C (GPU)");
    }
    
    // 打印性能信息
    printf("Matrix size: %d x %d\n", size, size);
    printf("GPU execution time: %f milliseconds\n", milliseconds);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    
    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}