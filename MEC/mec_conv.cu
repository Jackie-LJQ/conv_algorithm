#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "../common.h"
#include <time.h>
#include <sys/time.h>
#define TOTAL_RUN 3
#define MATMUL_BLOCKSIZE 8

void program(int, int,  int, int ,	int , int , int, int , int, int , size_t & );

int main() {
    //image width, height, channel, batch size
    int height = 224;
	int width = 224;
	int channels = 3;
	int batch_size = 1;//128;
    //kernel size, channel
	int ksize = 3; // 5-11
	int num_kernels = 16;
    //conv padding and stride
    int pad = 1; // 0-2
	int stride = 1; // 1
    struct timeval start;
    struct timeval stop;
    size_t used = 0;
    size_t totalMem = 0;
    double oneTime=0;
    double totalTime=0;
    program(1, 1, height, width, channels, batch_size, ksize, 
                num_kernels, pad, stride, used);
    std::fstream fperflog("perflog.csv", std::ios::out);
    fperflog << "numThread, blockSize, gridSize, avgTime, Memory" << std::endl;
    for (int blockSize=1; blockSize <= 2048; blockSize*=2) {
        //total number of thread < 2 * (number elements in outCol)
        unsigned int MAX_GRID_SIZE = (ksize * height * (width - ksize + 1)) / blockSize; 
        for (int gridSize=1; gridSize <= 2048; gridSize*=2) {
            if (gridSize >= MAX_GRID_SIZE) {
                continue;
            }
            totalTime = 0;
            totalMem = 0;
            for (int i=0; i < TOTAL_RUN; i++) {
                
                gettimeofday(&start, NULL);
                program(gridSize, blockSize, height, width, channels, batch_size, ksize, 
                num_kernels, pad, stride, used);
                gettimeofday(&stop, NULL);
                oneTime = (stop.tv_sec - start.tv_sec) * 1000.0;
                oneTime += (stop.tv_usec - start.tv_usec) / 1000.0;
                totalTime += oneTime;
                totalMem += used;
            }
            fperflog <<blockSize * gridSize << "," <<  blockSize << ","             
                                      << gridSize << "," << totalTime / TOTAL_RUN << "," 
                                      << used / (TOTAL_RUN *1e6) << std::endl;
            // break; //debug
        }
        // break; //debug
    }
    fperflog.close();
    return 0;
}


void program(int gridSize, int blockSize,  int height, int width,
	int channels, int batch_size, int ksize, int num_kernels, int pad, 
    int stride, size_t & used) {
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
    size_t avail;
    size_t total;
    cudaMalloc((void**) &gpu_colin.elements, sizeof(float)*gpu_colin.height * gpu_colin.width * gpu_colin.channels);  
    mec<<<gridSize, blockSize>>>(gpu_image, gpu_colin, ksize);
    cudaMemGetInfo(&avail, &total);
    used = total - avail;
    // printMatrix(image, "image");
    // Matrix cpu_colin;
    // cpu_colin.width = gpu_colin.width;
    // cpu_colin.height = gpu_colin.height;
    // cpu_colin.channels = gpu_colin.channels;
    // cpu_colin.batch_size = cpu_colin.batch_size;
    // transferFromDevice(gpu_colin, cpu_colin);
    // printMatrix(cpu_colin, "cpu_colin");
    cudaFree(gpu_image.elements);
    cudaFree(gpu_kernel.elements);
    cudaFree(gpu_colin.elements);
    free(image.elements);
    free(kernel.elements);
    // free(cpu_colin.elements);
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

// __global__ void matmul(Matrix gpu_colin, Matrix gpu_kernel) {
    
// }
