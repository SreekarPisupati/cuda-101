#include <stdio.h>

// Naive transpose: out[col][row] = in[row][col]
// The read is coalesced; the write is strided (uncoalesced).
__global__ void transpose(const float *in, float *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // global column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // global row
    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    size_t bytes = (size_t)width * height * sizeof(float);

    // 1. host memory
    float *in  = (float *)malloc(bytes);
    float *out = (float *)malloc(bytes);

    // 2. fill in[row][col] = row * width + col (so we can verify later)
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            in[row * width + col] = (float)(row * width + col);
        }
    }

    // 3. device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // 4. copy in -> device
    cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

    // 5. launch with a 2D block and 2D grid
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    transpose<<<grid, block>>>(d_in, d_out, width, height);

    // 6. copy out back to host
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    // 7. verify: out[col][row] must equal in[row][col]
    int errors = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (out[col * height + row] != in[row * width + col]) {
                errors++;
                if (errors <= 3) {
                    printf("MISMATCH at (row=%d, col=%d): got %f, expected %f\n",
                           row, col, out[col * height + row], in[row * width + col]);
                }
            }
        }
    }

    if (errors == 0) {
        printf("PASS: transpose of %dx%d matrix is correct\n", width, height);
    } else {
        printf("FAIL: %d mismatches\n", errors);
    }

    // 8. free everything
    free(in);
    free(out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
