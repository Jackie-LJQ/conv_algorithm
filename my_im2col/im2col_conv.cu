#include <iostream>
#include <stdlib.h>
#include "gen_img_kernel.h"

void im2colOnHost(Matrix &image, Matrix &colOutHost, int pad, int stride, int ksize);
//kernel functions
//num_kernels = output channel number
//gpu_image.channels = input channel number
// numWIndowPerRow = (gpu_image.width - ksize + 1) / stride
__global__ void im2col(Matrix gpu_image, Matrix gpu_colOut, int ksize, int stride) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx < gpu_colOut.height * gpu_colOut.width; idx += blockDim.x * gridDim.x) {
        int colOutIdy = idx / gpu_colOut.width; //index of slide window
        int colOutIdx = idx % gpu_colOut.width;
        int numWindowPerRow = (gpu_image.width - ksize + 1) / stride;
        int numWindowPerCol = (gpu_image.height - ksize + 1) / stride;
        int numWindowPerChannel = numWindowPerRow * numWindowPerCol;
        int windowIdy = colOutIdy / numWindowPerRow;
        int windowIdx = colOutIdy % numWindowPerRow;
        int channelIdy = colOutIdy / numWindowPerChannel;
        int eleInWindowIdy = colOutIdx / ksize;
        int eleInWindonIdx = colOutIdx % ksize;
        int inIdy = channelIdy * (ksize-1) + windowIdy + eleInWindowIdy; //-1 wrong in diff stride
        int inIdx = windowIdx + eleInWindonIdx;
        gpu_colOut.elements[colOutIdy * gpu_colOut.width + colOutIdx] = 
                            gpu_image.elements[inIdy*gpu_image.width + inIdx];
        // printf("idx: %d, %d\n", idx, gpu_image.elements[inIdy*gpu_image.width + inIdx]);
    }
}

//Host code
int main()
{
    //image width, height, channel, batch size
    const int height = 1024; //256
	const int width = 4; //256
	const int channels = 3;
	const int batch_size = 1;//128;
    //kernel size, channel
	const int ksize = 3; // 5-11
	const int num_kernels = 2;
    //conv padding and stride
    const int pad = 1; // 0-2
	const int stride = 1; // 1

    Matrix image;
    Matrix kernel;
    Matrix out;
    Matrix outHost;
    Matrix gpu_image;
    Matrix gpu_kernel;
    Matrix gpu_out;
    generate_data(image, kernel, gpu_out, height, width, channels, 
    batch_size, ksize, num_kernels, stride, pad);

    /*
    //For debug: serial result on host 
    im2colOnHost(image, outHost, pad, stride, ksize);    
    printMatrix(image, "image");
    printMatrix(outHost, "colOutHost"); 
    */

    transferToDevice(image, gpu_image);
    transferToDevice(kernel, gpu_kernel);

    Matrix gpu_Colout;
    gpu_Colout.width = ksize * ksize; //width of each row = kernel size
    
    int a = (width - ksize) / stride + 1;
    int b = (height - ksize) / stride + 1;
    gpu_Colout.height = a*b*channels ;//KERNEL_NUM
    gpu_Colout.channels = 1;
    gpu_Colout.batch_size = 1;
    cudaMalloc((void**) &gpu_Colout.elements, sizeof(float)*gpu_Colout.height * gpu_Colout.width);  
    int blockSize = 1; //ksize * ksize;
    int gridSize = 1; //gpu_Colout.height; 
    printMatrix(image, "image");
    printf("gpu_Out width: %d height: %d\n", gpu_Colout.width, gpu_Colout.height);
    im2col<<<gridSize, blockSize>>>(gpu_image, gpu_Colout, ksize, stride);


    Matrix cuColout;
    cuColout.width = gpu_Colout.width; //each row is of kernel size
    cuColout.height = gpu_Colout.height ;//KERNEL_NUM
    cuColout.channels = gpu_Colout.channels;
    cuColout.batch_size = gpu_Colout.batch_size;
    std::cout<<"\n";
    transferFromDevice(gpu_Colout, cuColout);
    // printMatrix(cuColout, "cuColout");

    // for (int i=0; i<cuColout.width * cuColout.height; i++) {
    //     if (cuColout.elements[i] != outHost.elements[i]) {
    //         std::cout<< "wrong in index: " << i << '\n';
    //     }
    // }

    // transferFromDevice(gpu_out, out);
    // printMatrix(out, "out");
    free_data(image, kernel);
    freeDevice(gpu_image, gpu_kernel, gpu_out, cuColout);
    return 0;
}

void im2colOnHost(Matrix &image, Matrix &colOutHost, int pad, int stride, int ksize) {
    std::cout<<"image: "<<image.width<<" " << image.height << " " << image.channels << "\n";
    int outWidth = (image.width - ksize) / stride + 1; //(image.width + 2*pad - ksize) / stride + 1;
    int outHeight = (image.height - ksize) / stride + 1; //(image.height + 2*pad - ksize) / stride + 1;
    int colOutHeight = outWidth * outHeight * image.channels;
    std::cout<<"out: "<<outWidth<<" " << outHeight << "\n";
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