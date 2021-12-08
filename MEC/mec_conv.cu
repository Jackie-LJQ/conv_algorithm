#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "../common.h"
#include <time.h>
#include <sys/time.h>
#define TOTAL_RUN 3
#define MATMUL_BLOCKSIZE 2

void program(int, int,  int, int ,	int , int , int, int , int, int , size_t & );
__global__ void blockMatrixMul(Matrix gpu_colin, Matrix gpu_kernel, Matrix gpu_colout);
__global__ void mec(Matrix gpu_image, Matrix gpu_colin, int ksize);

int main() {
    //image width, height, channel, batch size
    int height = 224;
	int width = 224;
	int channels = 3;
	int batch_size = 1;//128;
    //kernel size, channel
	int ksize = 3; // 5-11
	int num_kernels = 64;
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

    Matrix gpu_colout;
    gpu_colout.height = gpu_colin.width - ksize*ksize + 1;
    gpu_colout.width = gpu_colin.height * num_kernels;
    gpu_colout.channels = 1;
    gpu_colout.batch_size = 1;
    cudaMalloc((void**) &gpu_colout.elements, gpu_colout.width *
            gpu_colout.height * gpu_colout.channels * gpu_colout.batch_size * sizeof(float));
    
    dim3 dimBlock(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
    dim3 dimGrid(2,2);
    blockMatrixMul<<<dimBlock, dimGrid>>>(gpu_colin, gpu_kernel, gpu_colout);

    cudaMemGetInfo(&avail, &total);
    used = total - avail;
    /*
    printMatrix(image, "image");
    Matrix cpu_colin;
    cpu_colin.width = gpu_colin.width;
    cpu_colin.height = gpu_colin.height;
    cpu_colin.channels = gpu_colin.channels;
    cpu_colin.batch_size = cpu_colin.batch_size;
    transferFromDevice(gpu_colin, cpu_colin);
    printMatrix(cpu_colin, "cpu_colin");
    free(cpu_colin.elements);
    **/

    cudaFree(gpu_image.elements);
    cudaFree(gpu_kernel.elements);
    cudaFree(gpu_colin.elements);
    cudaFree(gpu_colout.elements);
    free(image.elements);
    free(kernel.elements);
    
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

__global__ void blockMatrixMul(Matrix gpu_colin, Matrix gpu_kernel, Matrix gpu_colout) {
    // coordinates of block
    int blockRow_k = blockIdx.y; //row index of kernel marix
    int blockRow_c = blockIdx.x; //row index of gpu_colin matrix 
    // printf("blockRow_k, blockRow_c: %d %d\n", blockRow_k, blockRow_c);
    //coordinates of element in block
    int row = threadIdx.y;
    int col = threadIdx.x; //(gpu_colin.width / MATMUL_BLOCKSIZE)
    int coloutIdx = blockRow_k * blockDim.y + row;
    int coloutIdy = blockRow_c * blockDim.x + col;
    for (int m=0; m < gpu_colin.width * gpu_colin.height/ MATMUL_BLOCKSIZE; m++) {
        __shared__ double As[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
        __shared__ double Bs[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
        int Aindy = blockRow_c * blockDim.x + row;
        int Aindx = m * blockDim.x + col;

        int Bindy = blockRow_k * blockDim.y + row;
        int Bindx = m * blockDim.x + col;        
        if (Aindx >= gpu_colin.width || Aindy >= gpu_colin.height || 
            Bindx >= gpu_colin.width || Bindy >= gpu_kernel.channels) {
                return;
            }
        As[row][col] = gpu_colin.elements[Aindy * gpu_colin.width + Aindx];
        Bs[row][col] = gpu_kernel.elements[Bindy * gpu_colin.width + Bindx];
        __syncthreads();
        // printf("Bs, %d %d,  %f \n", row, col, Bs[row][col]);
        // int breakPoint;
        // if (m==gpu_colin.width / MATMUL_BLOCKSIZE) {
        //     breakPoint = gpu_colin.width % MATMUL_BLOCKSIZE;
        // }
        // else {
        //     breakPoint = blockDim.x;
        // }
        for (int e=0; e < blockDim.x; e++) {
            gpu_colout.elements[
                coloutIdy * gpu_colout.width + coloutIdx] += As[row][e] * Bs[col][e];
            // printf("%d_%d, %d-%d, As  %d %d /%f, Bs %d %d /%f \n", 
            // coloutIdy, coloutIdx, blockRow_k, blockRow_c, 
            // row, e, As[row][e], col, e, Bs[col][e]);
        }  
        __syncthreads();      
    }
}