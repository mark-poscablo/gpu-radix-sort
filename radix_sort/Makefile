CUDA_PATH ?= "/usr/local/cuda-7.5"
NVCC := $(CUDA_PATH)/bin/nvcc
NVCC_OPTS=-O3 -arch=sm_37 -Xcompiler -Wall -Xcompiler -Wextra -m64

radix_sort: main.cu scan.o sort.o Makefile
	$(NVCC) -o radix_sort main.cu sort.o scan.o $(NVCC_OPTS)

sort.o: sort.cu
	$(NVCC) -c sort.cu $(NVCC_OPTS)

scan.o: scan.cu
	$(NVCC) -c scan.cu $(NVCC_OPTS)

clean:
	rm -f *.o radix_sort