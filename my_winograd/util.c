#include "util.h"
#include "math.h"

float* transpose(float* weight, int h, int w) {
  float* new_weight = (float*)malloc(w * h * 4);
  int i, j;
  for (i = 0; i < w; ++i) {
    for (j = 0; j < h; ++j) {
      new_weight[j * w + i] = weight[i * h + j];
    }
  }

  free(weight);
  return new_weight;
}

float* get_parameter(const char* filename, int size) {
  float* parameter = (float*)malloc(size * 4);
  if (!parameter) {
    printf("Bad Malloc\n");
    exit(0);
  }
  FILE* ptr = fopen(filename, "rb");

  if (!ptr) {
    printf("Bad file path: %p, %s\n", ptr, strerror(errno));
    exit(0);
  }
  fread(parameter, size * 4, 1, ptr);

  fclose(ptr);
  return parameter;
}
