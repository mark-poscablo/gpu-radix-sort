
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>

#include "scan.h"

void cpu_sort(unsigned int* h_out, unsigned int* h_in, size_t len)
{
	for (int i = 0; i < len; ++i)
	{
		h_out[i] = h_in[i];
	}

	std::sort(h_out, h_out + len);
}

__global__
void gpu_build_pred(unsigned int* const d_in,
	unsigned int* const d_preds,
	const size_t numElems,
	unsigned int bit_mask)
{
	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (glbl_t_idx >= numElems)
		return;

	unsigned int curr_elem = d_in[glbl_t_idx];
	// predicate is true if result is 0
	unsigned int pred = curr_elem & bit_mask;
	unsigned int pred_result = 0;
	if (pred == bit_mask)
	{
		pred_result = 1;
	}
	d_preds[glbl_t_idx] = pred_result;

	__syncthreads();

	unsigned int dummy = d_preds[glbl_t_idx];
}

__global__
void gpu_scatter_elems(unsigned int* const d_inputVals,
	unsigned int* const d_inputPos,
	unsigned int* const d_outputVals,
	unsigned int* const d_outputPos,
	const size_t numElems)
{

}

void radix_sort(unsigned int* const d_out,
	unsigned int* const d_in,
	unsigned int* const d_preds,
	unsigned int* const d_scanned_preds,
	const size_t numElems)
{
	unsigned int block_sz = 1024;
	unsigned int grid_sz = (unsigned int)ceil(float(numElems) / float(block_sz));

	// Do this for every bit, from LSB to MSB
	for (int sw = 0; sw < (sizeof(unsigned int) * 8); ++sw)
	{
		unsigned int bit_mask = 1 << sw;
		
		// Build predicate array
		gpu_build_pred<<<grid_sz, block_sz>>>(d_in, d_preds, numElems, bit_mask);

		// Scan predicate array
		//  If working with 0's, make sure the total sum is 
		//  recorded for the 1's

		// Scatter d_in's elements to their new locations in d_out
		//  Use predicate array to figure out which threads will move
		//  Use scanned predicate array to figure out the locations

	}	
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
	checkCudaErrors(cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice));

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
