#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Kernel128_winograd.h"
#include "util.h"

int main() {
  int num_test = 10, totalTime = 0, i;
  cudaSetDevice(0);

  for (i = 0; i < num_test; i++) {
    printf("---- Iter: %d ----\n", i);
    int res = -1;
    res = kernel_128();
    totalTime += res;
  }
  printf(
      "Average Total Time: %d us\n", totalTime / num_test);

  return 0;
}
