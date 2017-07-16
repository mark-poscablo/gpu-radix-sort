CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_37 -Xcompiler -Wall -Xcompiler -Wextra -m64

radix_sort: main.cu scan.o sort.o Makefile
	nvcc -o radix_sort main.cu sort.o scan.o $(NVCC_OPTS)

sort.o: sort.cu
	nvcc -c sort.cu $(NVCC_OPTS)

scan.o: scan.cu
	nvcc -c scan.cu $(NVCC_OPTS)

clean:
	rm -f *.o radix_sort
