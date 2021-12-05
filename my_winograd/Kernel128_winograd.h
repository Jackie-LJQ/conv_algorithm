#ifndef __KERNEL128_WINOGRAD_H__
#define __KERNEL128_WINOGRAD_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName128[] = "data/input_14_1_128.bin";
const char weight_winograd_Name128[] = "data/weight_winograd_128_128.bin";

int kernel_128();

#ifdef __cplusplus
}
#endif

#endif