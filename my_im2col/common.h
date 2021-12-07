#include <iostream>
#include <stdlib.h>

// Matrices are stored in row-major order:
// M(row, col) = M.elements[row * M.width + col]
typedef struct {
    int width;
    int height;
    int channels;
    int batch_size;
    float* elements;
} Matrix;

void printMatrix(Matrix &, std::string);

void init_data(Matrix &data, int size) {
    for (int i=0; i<size; i++) {
        data.elements[i] = (float) i;
    }
}

void generate_data(Matrix &image, Matrix &kernel, int height, int width,
            int channels, int batch_size, int ksize, int num_kernels, int pad, int stride) {
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.batch_size = batch_size;
    int imageSize = height*width*channels*batch_size;
    image.elements = (float*)malloc(imageSize * channels * sizeof(float));
    init_data(image, imageSize);

    kernel.width = ksize;
    kernel.height = ksize;
    kernel.channels = num_kernels;
    kernel.batch_size = 1;
    int kernelSize = ksize * ksize * num_kernels;
    kernel.elements = (float*) malloc(kernelSize * sizeof(float));
    init_data(kernel, kernelSize);
    printMatrix(kernel, "kernel");
}


void transferToDevice(Matrix &a, Matrix &gpu_a) {
    gpu_a.width = a.width;
    gpu_a.height = a.height;
    gpu_a.channels = a.channels;
    gpu_a.batch_size = a.batch_size;
    int aSize = a.width * a.height * a.batch_size * a.channels ;
    cudaMalloc((void**) &gpu_a.elements, aSize * sizeof(float));
    cudaMemcpy(gpu_a.elements, a.elements, sizeof(float) * aSize, cudaMemcpyHostToDevice);
}

void transferFromDevice(Matrix &gpu_out, Matrix &out){
    out.width = gpu_out.width;
    out.height = gpu_out.height;
    out.channels = gpu_out.channels;
    out.batch_size = gpu_out.batch_size;
    int outSize = gpu_out.width * gpu_out.height * gpu_out.batch_size * gpu_out.channels ;
    out.elements = (float*) malloc(outSize * sizeof(float));
    cudaMemcpy(out.elements, gpu_out.elements, sizeof(float) * outSize, cudaMemcpyDeviceToHost);
}


// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.width + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.width + col] = value;
}


void printMatrix(Matrix &A, std::string name) {
    std::cout << "\n\nPrint matrix " << name;
    for (int i=0; i<A.width*A.height*A.channels; i++) {
        if (i%A.width==0) {
            std::cout<<"\n";
        }
        std::cout << A.elements[i] << " ";
    }
    std::cout << "\n";
}

// __global__ void blockMatrixMul(Matrix gpu_colin, Matrix gpu_kernel, Matrix gpu_colout) {
//     // coordinates of block
//     int blockRow = blockIdx.y;
//     int blockCol = blockIdx.x;
//     //coordinates of element in block
//     int row = threadIdx.y;
//     int col = threadIdx.x;
//     for (int m=0; m < (gpu_colin.height / MATMUL_BLOCKSIZE); m++) {
//         __shared__ float As[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
//         __shared__ float Bs[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
//         int Aindy = blockRow * blockDim.y + row;
//         int Aindx = m * blockDim.x + col;
//         As[row][col] = gpu_colin.elements[Aindy * gpu_colin.height + Aindx];
//         int Bindy = m * blockDim.y + row;
//         int Bindx = blockCol * blockDim.x + col;
//         Bs[row][col] = gpu_kernel.elements[Bindy * gpu_kernel.width + Bindx];
//         __syncthreads();
//         for (int e=0; e < blockDim.x; e++) {
//             gpu_colout.elements[Aindy * gpu_colout.width + Bindx] += As[row][e] * Bs[e][col];
//         }        
//     }

// }