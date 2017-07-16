CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

radix_sort: scan.o sort.o Makefile
	nvcc -o radix_sort main.cu sort.o scan.o $(NVCC_OPTS)

sort.o:
	nvcc -c sort.cu $(NVCC_OPTS)

scan.o:
	nvcc -c scan.cu $(NVCC_OPTS)

clean:
	rm -f *.o radix_sort
