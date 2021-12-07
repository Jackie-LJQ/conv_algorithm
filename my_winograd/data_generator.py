from __future__ import print_function
import sys
import numpy as np
import re
import time
import difflib
import array
import requests
import os
import shutil
import scipy.spatial.distance as spd
import numpy as np
from numpy.random import *
from random import randint
import random
#import matplotlib.pyplot as plt
from scipy import misc

def input_generator(input_channel = 128, feature_map_size = 14, padding = 1):
	parameters = (feature_map_size + 2*padding)*(feature_map_size + 2*padding) * input_channel
	a = (np.array(rand(parameters))-0.5).astype(np.float32)
	des = open("data/input_" + str(feature_map_size) + '_' + str(padding) + '_' + str(input_channel) + ".bin", "wb")
	des.write(a)

def weight_generator(input_channel = 128, output_channel = 128):
	parameters = input_channel*output_channel * 3*3
	in_ = (np.array(rand(parameters))-0.5).astype(np.float32)

	### Weights_Winograd
	in_ = in_.reshape(input_channel*output_channel, 3,3)
	G = np.array([[0.25,0,0], [-1.0/6,-1.0/6,-1.0/6], [-1.0/6,1.0/6,-1.0/6], [1.0/24,1.0/12,1.0/6], [1.0/24,-1.0/12,1.0/6], [0,0,1]])

	out_ = [0] * input_channel*output_channel * 6*6
	for i in range(output_channel):
		for j in range(input_channel):
			b = np.dot(G, in_[i*input_channel+j])
			b = np.dot(b, G.transpose())
			offset = j*output_channel+i
			for x in range(6):
				for y in range(6):
					out_[((x*6+y) * input_channel*output_channel) + offset] = b[x][y]

	des = open("data/weight_winograd_" + str(input_channel) + '_' + str(output_channel) + ".bin", "wb")
	des.write(np.array(out_).astype(np.float32))


if __name__ == '__main__':
	input_generator(input_channel = 128)
	print('Input generated')

	weight_generator(128, 128)
	print('Weights generated')