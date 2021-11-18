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

void init_data(float *data, int size) {
    for (int i=0; i<size; i++) {
        data[i] = (float) i;
    }
}

void generate_data(Matrix image, Matrix kernel, int height, int width,
            int channels, int batch_size, int ksize, int num_kernels) {
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.batch_size = batch_size;
    int imageSize = height*width*channels*batch_size;
    image.elements = (float*)malloc(imageSize * sizeof(float));
    init_data(image.elements, imageSize);

    kernel.width = ksize;
    kernel.height = ksize;
    kernel.channels = num_kernels;
    int kernelSize = ksize * ksize * num_kernels;
    kernel.elements = (float*) malloc(kernelSize * sizeof(float));
    init_data(kernel.elements, kernelSize);
}

void free_data(Matrix image, Matrix kernel) {
    free(image.elements);
    free(kernel.elements);
}