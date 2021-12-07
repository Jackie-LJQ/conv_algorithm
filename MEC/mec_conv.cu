#include <iostream>
#include <stdlib.h>
#include "../common.h"
#include <time.h>
#define TOTAL_RUN 1
#define MATMUL_BLOCKSIZE 2

__global__ void mec(Matrix , Matrix , int );

int main() {
    //image width, height, channel, batch size
    int height = 4;
	int width = 4;
	int channels = 1;
	int batch_size = 1;//128;
    //kernel size, channel
	int ksize = 3; // 5-11
	int num_kernels = 4;
    //conv padding and stride
    int pad = 1; // 0-2
	int stride = 1; // 1
    Matrix image;
    Matrix kernel;
    Matrix gpu_image;
    Matrix gpu_kernel;
    generate_data(image, kernel, height, width, channels, 
                        batch_size, ksize, num_kernels, stride, pad);

    transferToDevice(image, gpu_image);
    transferToDevice(kernel, gpu_kernel);
    Matrix gpu_colin;
    gpu_colin.width = ksize * image.height;
    gpu_colin.height = image.width - ksize + 1;
    gpu_colin.channels = 1;
    gpu_colin.batch_size = 1;
    cudaMalloc((void**) &gpu_colin.elements, sizeof(float)*gpu_colin.height * gpu_colin.width * gpu_colin.channels);  
    mec<<<2, 3>>>(gpu_image, gpu_colin, ksize);
    printMatrix(image, "image");
    Matrix cpu_colin;
    cpu_colin.width = gpu_colin.width;
    cpu_colin.height = gpu_colin.height;
    cpu_colin.channels = gpu_colin.channels;
    cpu_colin.batch_size = cpu_colin.batch_size;
    transferFromDevice(gpu_colin, cpu_colin);
    printMatrix(cpu_colin, "cpu_colin");
    cudaFree(gpu_image.elements);
    cudaFree(gpu_kernel.elements);
    cudaFree(gpu_colin.elements);
    free(image.elements);
    free(kernel.elements);
    free(cpu_colin.elements);
}

__global__ void mec(Matrix gpu_image, Matrix gpu_colin, int ksize) {
    for (int idx=blockIdx.x * blockDim.x + threadIdx.x; idx < 
    gpu_colin.height * gpu_colin.width; idx += blockDim.x * gridDim.x) {
        int colinIdy = idx / gpu_colin.width; //patch index
        int colinIdx = idx % gpu_colin.width;
        int imgIdx = colinIdy + colinIdx % ksize;
        int imgIdy = colinIdx / ksize;
        gpu_colin.elements[colinIdy * gpu_colin.width + colinIdx] = 
            gpu_image.elements[imgIdy * gpu_image.width + imgIdx];


    }
}
