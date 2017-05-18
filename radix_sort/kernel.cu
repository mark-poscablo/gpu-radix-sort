
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>

void cpu_sort(unsigned int* h_out, unsigned int* h_in, size_t len)
{
	for (int i = 0; i < len; ++i)
	{
		h_out[i] = h_in[i];
	}

	std::sort(h_out, h_out + len);
}

__global__
void gpu_build_pred(unsigned int* const d_inputVals,
	unsigned int* const d_inputPos,
	unsigned int* const d_outputVals,
	unsigned int* const d_outputPos,
	const size_t numElems)
{

}

__global__
void gpu_scatter_elems(unsigned int* const d_inputVals,
	unsigned int* const d_inputPos,
	unsigned int* const d_outputVals,
	unsigned int* const d_outputPos,
	const size_t numElems)
{

}

void radix_sort(unsigned int* const d_inputVals,
	unsigned int* const d_inputPos,
	unsigned int* const d_outputVals,
	unsigned int* const d_outputPos,
	const size_t numElems)
{
	// Build predicate array

	// Scan predicate array
	//  If working with 0's, make sure the total sum is 
	//  recorded for the 1's

	// Scatter d_in's elements to their new locations in d_out
	//  Use predicate array to figure out which threads will move
	//  Use scanned predicate array to figure out the locations
	
	// Do this for every bit, from LSB to MSB
}

int main()
{
	unsigned int* h_in = new unsigned int[1024];
	unsigned int* h_out = new unsigned int[1024];

	for (int i = 0; i < 1024; i++)
	{
		h_in[i] = 1023 - i;
		//std::cout << h_in[i] << " ";
	}

	cpu_sort(h_out, h_in, 1024);
	for (int i = 0; i < 1024; i++)
	{
		//std::cout << h_in[i] << " ";
		std::cout << h_out[i] << " ";
	}

	delete[] h_out;
	delete[] h_in;
}
