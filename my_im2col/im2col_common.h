#define TOTAL_RUN 3
#define MATMUL_BLOCKSIZE 8
// void im2colOnHost(Matrix&, Matrix&, int, int, int);
__global__ void im2col(Matrix, Matrix, int, int, int, int, int);
__global__ void blockMatrixMul(Matrix , Matrix , Matrix );
int program(int gridSize, int blockSize,  int height, int width,
	int channels, int batch_size, int ksize, int num_kernels, int pad, int stride, size_t & used);
