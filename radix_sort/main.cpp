#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>

#include "sort.h"
#include "utils.h"

void cpu_sort(unsigned int* h_out, unsigned int* h_in, size_t len)
{
	for (int i = 0; i < len; ++i)
	{
		h_out[i] = h_in[i];
	}

	std::sort(h_out, h_out + len);
}

int main()
{
	unsigned int num_elems = 1024;

	unsigned int* h_in = new unsigned int[num_elems];
	unsigned int* h_out_cpu = new unsigned int[num_elems];
	unsigned int* h_out_gpu = new unsigned int[num_elems];

	for (int i = 0; i < num_elems; i++)
	{
		h_in[i] = (num_elems - 1) - i;
		//std::cout << h_in[i] << " ";
	}

	unsigned int* d_in;
	unsigned int* d_preds;
	unsigned int* d_scanned_preds;
	unsigned int* d_out;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * num_elems));
	checkCudaErrors(cudaMalloc(&d_preds, sizeof(unsigned int) * num_elems));
	checkCudaErrors(cudaMalloc(&d_scanned_preds, sizeof(unsigned int) * num_elems));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * num_elems));
	checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice));
	radix_sort(d_out, d_in, d_preds, d_scanned_preds, num_elems);
	checkCudaErrors(cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_scanned_preds));
	checkCudaErrors(cudaFree(d_preds));
	checkCudaErrors(cudaFree(d_in));

	cpu_sort(h_out_cpu, h_in, num_elems);
	for (int i = 0; i < num_elems; i++)
	{
		//std::cout << h_in[i] << " ";
		std::cout << h_out_cpu[i] << " ";
	}

	delete[] h_out_gpu;
	delete[] h_out_cpu;
	delete[] h_in;
}
