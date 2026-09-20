#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = (1 << 20) - 1;                          // pick a size, e.g. 1<<20 (≈1 million)
    size_t bytes = N * sizeof(float);

    // 1. host arrays (malloc)
    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);
    // 2. fill A and B with some values
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    // 3. device arrays (cudaMalloc)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    // 4. copy A, B host→device
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
    // 5. launch kernel:  vectorAdd<<<numBlocks, threadsPerBlock>>>(...)
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // 6. copy C device→host
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    // 7. verify + print PASS/FAIL
    for (int i = 0; i < N; i++) {
        if (C[i] != 3.0f) {
            printf("FAIL: C[%d] = %f\n", i, C[i]);
            return -1;
        }
    }
    printf("PASS\n");   
    // 8. free everything
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    

    return 0;
}
