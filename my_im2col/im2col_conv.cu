#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "common.h"
#include <time.h>
#include <sys/time.h>
#define TOTAL_RUN 5

void im2colOnHost(Matrix&, Matrix&, int, int, int);
__global__ void im2col(Matrix, Matrix, int, int, int, int, int);

//Host code
int program(int gridSize, int blockSize,  int height, int width,
	int channels, int batch_size, int ksize, int num_kernels, int pad, int stride)
{
    Matrix image;
    Matrix kernel;
    Matrix outHost;
    Matrix gpu_image;
    Matrix gpu_kernel;
    generate_data(image, kernel, height, width, channels, 
                        batch_size, ksize, num_kernels, stride, pad);

    // /*
    //For debug: serial result on host 
    im2colOnHost(image, outHost, pad, stride, ksize);    
    // printMatrix(image, "image");
    // printMatrix(outHost, "colOutHost"); 
    // */

    transferToDevice(image, gpu_image);
    transferToDevice(kernel, gpu_kernel);

    Matrix gpu_Colout;
    gpu_Colout.width = ksize * ksize; //width of each row = kernel size
    int numWindowPerRow = (gpu_image.width - ksize) / stride + 1;
    int numWindowPerCol = (gpu_image.height - ksize) / stride + 1;
    int numWindowPerChannel = numWindowPerRow * numWindowPerCol; //number of lside window in each image channel
    int kernelNum = numWindowPerChannel * channels;
    gpu_Colout.height = kernelNum; //KERNEL_NUM
    gpu_Colout.channels = 1;
    gpu_Colout.batch_size = 1;
    cudaMalloc((void**) &gpu_Colout.elements, sizeof(float)*gpu_Colout.height * gpu_Colout.width * gpu_Colout.channels);  
    im2col<<<gridSize, blockSize>>>(gpu_image, gpu_Colout, ksize, 
                            stride, numWindowPerRow, numWindowPerCol, numWindowPerChannel);
    Matrix colOutDev;
    colOutDev.width = gpu_Colout.width; //each row is of kernel size
    colOutDev.height = gpu_Colout.height ;//KERNEL_NUM
    colOutDev.channels = gpu_Colout.channels;
    colOutDev.batch_size = gpu_Colout.batch_size;
    std::cout<<"\n";
    transferFromDevice(gpu_Colout, colOutDev);
    // printMatrix(colOutDev, "colOutDev");

    for (int i=0; i<colOutDev.width * colOutDev.height; i++) {
        if (colOutDev.elements[i] != outHost.elements[i]) {
            std::cout<< "wrong in index: " << i << '\n';
        }
    }

    // printMatrix(out, "out");
    cudaFree(gpu_image.elements);
    cudaFree(gpu_kernel.elements);
    cudaFree(gpu_Colout.elements);
    free(image.elements);
    free(kernel.elements);
    free(outHost.elements);
    free(colOutDev.elements);
    return 0;
}


int main() {
    //image width, height, channel, batch size
    int height = 256;
	int width = 256;
	int channels = 80;
	int batch_size = 1;//128;
    //kernel size, channel
	int ksize = 3; // 5-11
	int num_kernels = 2;
    //conv padding and stride
    int pad = 1; // 0-2
	int stride = 1; // 1
    struct timeval start;
    struct timeval stop;
    double oneTime=0;
    double totalTime=0;
    int numWindowPerRow = (width - ksize) / stride + 1;
    int numWindowPerCol = (height - ksize) / stride + 1;
    int numWindowPerChannel = numWindowPerRow * numWindowPerCol; //number of lside window in each image channel
    int kernelNum = numWindowPerChannel * channels;
    //Mesure effect of different block size
    // printf("blockSize, gridSize, avgTime\n");
    std::fstream fperflog("perflog.csv", std::ios::out);
    fperflog << "numThread, blockSize, gridSize, avgTime" << std::endl;
    for (int blockSize=1; blockSize <= 2048; blockSize*=2) {
        //total number of thread < 2 * (number elements in outCol)
        unsigned int MAX_GRID_SIZE = (kernelNum + blockSize - 1) / blockSize; 
        for (int gridSize=1; gridSize <= 2048; gridSize*=2) {
            if (gridSize >= MAX_GRID_SIZE) {
                continue;
            }
            totalTime = 0;
            for (int i=0; i < TOTAL_RUN; i++) {
                gettimeofday(&start, NULL);
                program(gridSize, blockSize, height, width, channels, batch_size, ksize, 
                num_kernels, pad, stride);
                gettimeofday(&stop, NULL);
                oneTime = (stop.tv_sec - start.tv_sec) * 1000.0;
                oneTime += (stop.tv_usec - start.tv_usec) / 1000.0;
                totalTime += oneTime;
            }
            fperflog <<blockSize * gridSize << "," <<  blockSize << ","             
                                      << gridSize << "," << totalTime / TOTAL_RUN << std::endl;
        }
    }
    fperflog.close();
}
void im2colOnHost(Matrix &image, Matrix &colOutHost, int pad, int stride, int ksize) {
    // std::cout<<"image: "<<image.width<<" " << image.height << " " << image.channels << "\n";
    int outWidth = (image.width - ksize) / stride + 1; //(image.width + 2*pad - ksize) / stride + 1;
    int outHeight = (image.height - ksize) / stride + 1; //(image.height + 2*pad - ksize) / stride + 1;
    int colOutHeight = outWidth * outHeight * image.channels;
    // std::cout<<"out: "<<outWidth<<" " << outHeight << "\n";
    colOutHost.height = colOutHeight;
    colOutHost.width = ksize * ksize;
    colOutHost.channels = 1;
    colOutHost.batch_size = 1;
    colOutHost.elements = (float*) malloc(colOutHeight*ksize*ksize*sizeof(float));
    int colOutIdy = 0;
    for (int channelId=0; channelId < image.channels; channelId++) {
        for (int rowIdx=0; rowIdx < image.height-ksize+1; rowIdx++) {  
            for (int colIdx=0; colIdx < image.width-ksize+1; colIdx++) {  
                for (int kernelY=0; kernelY < ksize; kernelY++) {        
                    for (int kernelX=0; kernelX < ksize; kernelX++) {                                            
                        int colOutIdx = kernelY * ksize + kernelX;
                        int inIdx = colIdx + kernelX;
                        int inIdy = channelId * image.height + rowIdx + kernelY;
                        // std::cout<<image.elements[inIdy * image.width + inIdx]<<" ";
                        colOutHost.elements[colOutIdy * ksize*ksize + colOutIdx] = image.elements[inIdy * image.width + inIdx];
                    }
                }
                colOutIdy += 1; //append to the last of colOutHost
            }
        }
    }
    // std::cout << "\n";
}


//kernel functions
__global__ void im2col(Matrix gpu_image, Matrix gpu_colOut, int ksize, int stride, 
                            int numWindowPerRow, int numWindowPerCol, int numWindowPerChannel) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx < gpu_colOut.height * gpu_colOut.width; idx += blockDim.x * gridDim.x) {
        int colOutIdy = idx / gpu_colOut.width; //index of slide window
        int colOutIdx = idx % gpu_colOut.width;
        int windowIdy = colOutIdy / numWindowPerRow;
        int windowIdx = colOutIdy % numWindowPerRow;
        int channelIdy = colOutIdy / numWindowPerChannel;
        int eleInWindowIdy = colOutIdx / ksize;
        int eleInWindonIdx = colOutIdx % ksize;
        int inIdy = channelIdy * (ksize-1) + windowIdy + eleInWindowIdy; //Todo: -1 wrong in diff stride
        int inIdx = windowIdx + eleInWindonIdx;
        gpu_colOut.elements[colOutIdy * gpu_colOut.width + colOutIdx] = 
                            gpu_image.elements[inIdy*gpu_image.width + inIdx];
    }
}