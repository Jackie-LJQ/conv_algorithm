#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <time.h>

#include "util.h"
#include "Kernel128_winograd.h"


#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

#define MY_KERNEL 1

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] )

__global__ void kernel_128_winograd_BtdB(float *pInputs, float *pOutputs) {
	int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Iny1 = threadIdx.y, Inz = threadIdx.x;
	int Iny = Iny0+Iny1, stride_r = 2048, stride_c = 128; // 2048 = 16*128
	int c_glb_start = Inx*stride_r + Iny*stride_c + Inz, c_input = Iny1*stride_c + Inz;

	extern __shared__ float input[];

	int tmp[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
	for (int i = 0; i < 6; i++) {
		input[c_input + tmp[i]] = pInputs[c_glb_start + i*stride_r];
	}
	__syncthreads();

	float BTd[6];
	switch(Iny1) {
		case 0:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 0, j, Inz)*4 - d(input, 2, j, Inz)*5 + d(input, 4, j, Inz);
			}
			break;
		case 1:
			for (int j = 0; j < 6; j++) {
				BTd[j] = -d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 + d(input, 3, j, Inz) + d(input, 4, j, Inz);
			}
			break;
		case 2:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 - d(input, 3, j, Inz) + d(input, 4, j, Inz);
			}
			break;
		case 3:
			for (int j = 0; j < 6; j++) {
				BTd[j] = -d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) + d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
			}
			break;
		case 4:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) - d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
			}
			break;
		case 5:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*4 - d(input, 3, j, Inz)*5 + d(input, 5, j, Inz);
			}
			break;
	}
	__syncthreads();

	int tmp_offset = Iny1*768+Inz;
	for (int i = 0; i < 6; i++) {
		input[tmp_offset + i*stride_c] = BTd[i];
	}
	__syncthreads();

	float BTdB[6];
	switch(Iny1) {
		case 0:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 0, Inz) - 5*d(input, i, 2, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 1:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = -4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) + d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 2:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) - d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 3:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = -2*d(input, i, 1, Inz) - d(input, i, 2, Inz) + 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 4:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 2*d(input, i, 1, Inz) - d(input, i, 2, Inz) - 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 5:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 1, Inz) - 5*d(input, i, 3, Inz) + d(input, i, 5, Inz);
			}
			break;
	}
	__syncthreads();

	for (int i = 0; i < 6; i++) {
		pOutputs[(Iny1 + i*6)*2048 + (blockIdx.x*4+blockIdx.y)*128 + Inz] = BTdB[i];
	}
}


__global__ void kernel_128_OuterProduct_128(float *A, float *B, float *C) {
	int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
	int c_input = tY*128 + tX, c_kernel = c_input, T_offset = (Tile<<11) + (Part<<10) + c_input, B_offset = (Tile<<14) + c_kernel;
	
	extern __shared__ float input[];
	float *kernel = input + 1024, *out = kernel + 8192;
	int B_stride[32] = {0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968};//, 4096, 4224, 4352, 4480, 4608, 4736, 4864, 4992, 5120, 5248, 5376, 5504, 5632, 5760, 5888, 6016, 6144, 6272, 6400, 6528, 6656, 6784, 6912, 7040, 7168, 7296, 7424, 7552, 7680, 7808, 7936, 8064};
	out[c_input] = 0.0f;

	input[c_input] = A[T_offset];

	for (int k = 0; k < 4; k++) {
		int B_start = B_offset + (k<<12); // 32*64
		kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
		kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
		__syncthreads();

		float sum = 0;
		int y_tmp = (tY<<7)+(k<<5);
		for (int j = 0; j < 32; j++) {
			sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
		}
		out[tY*128 + tX] += sum;
		__syncthreads();
	}

	C[T_offset] = out[c_input];
}

int kernel_128() {
	float *input_ = get_parameter(inputName128, 16*16*128);
	float *input, *output, *l_weights;
	struct timespec start, stop;
    double runtime;
	cudaError_t s;

	/////////////////////////////////

	// My Kernel

	/////////////////////////////////


	/*  1. Data preparation  */
	float *t_input, *ip;
	float *kernel = get_parameter(weight_winograd_Name128, 36*128*128);

	int nInput = 16*16*128, nOutput = 16*16*128, nWeights = 36*128*128, nTransInput = 16*6*6*128, nInnerProd = 16*6*6*128;
	cudaMalloc((void **) &input, nInput<<3);
	cudaMalloc((void **) &output, nOutput<<2);
	cudaMalloc((void **) &l_weights, nWeights<<2);
	cudaMalloc((void **) &t_input, nTransInput<<2);
	cudaMalloc((void **) &ip, nInnerProd<<2);
	cudaMemset((void *) input, 0, nInput<<3);
	cudaMemset((void *) output, 0, nOutput<<2);
	cudaMemset((void *) t_input, 0, nTransInput<<2);
	cudaMemset((void *) l_weights, 0, nWeights<<2);
	cudaMemset((void *) ip, 0, nInnerProd<<2);
	cudaMemcpy(input, input_, nInput<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_weights, kernel, nWeights<<2, cudaMemcpyHostToDevice);
	
	float tmp_winograd[nOutput];

	
	/*  2. Computing  */
	clock_gettime(CLOCK_REALTIME, &start);

	kernel_128_winograd_BtdB <<<dim3(4, 4), dim3(128, 6), (6*6*128)<<2 >>> (input, t_input);
	kernel_128_OuterProduct_128<<<dim3(36, 2), dim3(128, 8), (8*128 + 64*128 + 8*128)<<2 >>> (t_input, l_weights, ip);
	//cudaCheckError();
	cudaDeviceSynchronize();
	
	clock_gettime(CLOCK_REALTIME, &stop);
	runtime = (stop.tv_sec - start.tv_sec) * 1e9 + (double)(stop.tv_nsec - start.tv_nsec);
	printf("Runtime = %d us\n", runtime); 


	/*  3. Copy back and free  */
	s = cudaMemcpy(tmp_winograd, output, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	//cudaCheckError();

	cudaFree(t_input);
	cudaFree(output);
	cudaFree(l_weights);
	cudaFree(ip);

	free(kernel);

	return runtime;
}