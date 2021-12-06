CC=gcc
CPP=g++
AR=ar
NVCC=nvcc

CSRCS := $(shell find . -name '*.c' -not -name '._*')
COBJS := $(subst .c,.o,$(CSRCS))

CUSRCS := $(shell find . -name '*.cu' -not -name '._*')
CUOBJS := $(subst .cu,.o,$(CUSRCS))

CUFLAGS= \
-I. \
-Xcompiler \
-fPIC

LDFLAGS=-L. -lm -lpthread -lrt

all: im2col

%.o: %.c
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $(basename $@).o

im2col: $(CUOBJS) $(COBJS)
	$(NVCC) -o im2col $(CUOBJS) $(COBJS) $(LIBDIR) $(LDFLAGS)

clean:
	find . -name "*.o" -exec rm -f '{}' ';'
	rm -f im2col