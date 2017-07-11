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
	// Set up clock for timing comparisons
	srand(time(NULL));
	std::clock_t start;
	double duration;

	for (int i = 0; i < 28; ++i)
	{
		unsigned int num_elems = (1 << i) - 1;
		//unsigned int num_elems = 8192;
		std::cout << "h_in size: " << num_elems << std::endl;

		unsigned int* h_in = new unsigned int[num_elems];
		unsigned int* h_out_cpu = new unsigned int[num_elems];
		unsigned int* h_out_gpu = new unsigned int[num_elems];

		for (int i = 0; i < num_elems; i++)
		{
			h_in[i] = (num_elems - 1) - i;
			//std::cout << h_in[i] << " ";
		}
		start = std::clock();
		cpu_sort(h_out_cpu, h_in, num_elems);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "CPU time: " << duration << std::endl;

		unsigned int* d_in;
		unsigned int* d_preds;
		unsigned int* d_scanned_preds;
		unsigned int* d_out;
		checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * num_elems));
		checkCudaErrors(cudaMalloc(&d_preds, sizeof(unsigned int) * num_elems));
		checkCudaErrors(cudaMalloc(&d_scanned_preds, sizeof(unsigned int) * num_elems));
		checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * num_elems));
		checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice));
		start = std::clock();
		radix_sort(d_out, d_in, num_elems);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "GPU time: " << duration << std::endl;
		checkCudaErrors(cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_out));
		checkCudaErrors(cudaFree(d_scanned_preds));
		checkCudaErrors(cudaFree(d_preds));
		checkCudaErrors(cudaFree(d_in));

		// Check for any mismatches between outputs of CPU and GPU
		bool match = true;
		int index_diff = 0;
		for (int i = 0; i < num_elems; ++i)
		{
			if (h_out_cpu[i] != h_out_gpu[i])
			{
				match = false;
				index_diff = i;
				break;
			}
		}
		std::cout << "Match: " << match << std::endl;

		// Detail the mismatch if any
		if (!match)
		{
			std::cout << "Difference in index: " << index_diff << std::endl;
			std::cout << "CPU: " << h_out_cpu[index_diff] << std::endl;
			std::cout << "GPU Radix Sort: " << h_out_gpu[index_diff] << std::endl;
			int window_sz = 10;

			std::cout << "Contents: " << std::endl;
			std::cout << "CPU: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << h_out_cpu[index_diff + i] << ", ";
			}
			std::cout << std::endl;
			std::cout << "GPU Radix Sort: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << h_out_gpu[index_diff + i] << ", ";
			}
			std::cout << std::endl;
		}

		delete[] h_out_gpu;
		delete[] h_out_cpu;
		delete[] h_in;

		std::cout << std::endl;
	}
}
