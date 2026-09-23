#include <stdio.h>

#define TILE_DIM 32

// Naive transpose (milestone 3) — kept here to compare speed.
__global__ void transpose_naive(const float *in, float *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

// Tiled transpose: shared memory + __syncthreads, coalesced on BOTH sides.
// The +1 padding on the tile avoids shared-memory bank conflicts.
__global__ void transpose_tiled(const float *in, float *out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Phase 1: cooperative load (coalesced read from global memory)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    // Phase 2: transposed write (coalesced write to global memory)
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x_out < height && y_out < width) {
        out[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int width = 2048;
    int height = 2048;
    size_t bytes = (size_t)width * height * sizeof(float);

    float *in  = (float *)malloc(bytes);
    float *out = (float *)malloc(bytes);
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
            in[row * width + col] = (float)(row * width + col);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- time naive ----
    cudaEventRecord(start);
    transpose_naive<<<grid, block>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t_naive = 0.0f;
    cudaEventElapsedTime(&t_naive, start, stop);

    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);
    int err_naive = 0;
    for (int row = 0; row < height && !err_naive; row++)
        for (int col = 0; col < width; col++)
            if (out[col * height + row] != in[row * width + col]) { err_naive = 1; break; }

    // ---- time tiled ----
    cudaEventRecord(start);
    transpose_tiled<<<grid, block>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t_tiled = 0.0f;
    cudaEventElapsedTime(&t_tiled, start, stop);

    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);
    int err_tiled = 0;
    for (int row = 0; row < height && !err_tiled; row++)
        for (int col = 0; col < width; col++)
            if (out[col * height + row] != in[row * width + col]) { err_tiled = 1; break; }

    printf("naive : %s  %.3f ms\n", err_naive ? "FAIL" : "PASS", t_naive);
    printf("tiled : %s  %.3f ms\n", err_tiled ? "FAIL" : "PASS", t_tiled);
    printf("speedup: %.2fx\n", t_naive / t_tiled);

    free(in); free(out);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
