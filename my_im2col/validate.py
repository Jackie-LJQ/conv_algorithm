import numpy as np
from scipy import signal

height, width, channels = 4, 4, 1
ksize, num_kernel = 3, 4
input = np.empty((channels, height, width))
kernel = np.empty((num_kernel, ksize, ksize))

def gen_data(data):
    tmp = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data[i][j][k] = tmp
                tmp+=1
    # print(np.ravel(data))

gen_data(input)
gen_data(kernel)
res = np.empty((num_kernel, height-ksize+1, width-ksize+1))
for index in range(num_kernel):
    res[index] = signal.convolve2d(input[0], kernel[index], mode="valid")

print(input)

print("##########################\n")
n_kernel = np.reshape(kernel, (-1, ksize *ksize))
print(n_kernel)
print("##########################\n")
print(res)