#include "sort.h"

__global__
void gpu_build_pred(unsigned int* const d_out,
	unsigned int* const d_in,
	const size_t numElems,
	unsigned int bit_mask,
	unsigned int zero_or_one)
{
	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (glbl_t_idx >= numElems)
		return;

	unsigned int curr_elem = d_in[glbl_t_idx];
	// predicate is true if result is 0
	unsigned int pred = curr_elem & bit_mask;
	unsigned int pred_result = zero_or_one ? 0 : 1;
	if (pred == bit_mask)
	{
		pred_result = zero_or_one ? 1 : 0;
	}
	d_out[glbl_t_idx] = pred_result;

	__syncthreads();

	unsigned int dummy = d_out[glbl_t_idx];
}

__global__
void gpu_scatter_elems(unsigned int* const d_out,
	unsigned int* const d_in,
	unsigned int* const d_preds,
	unsigned int* const d_scanned_preds,
	unsigned int* const d_out_offset,
	const size_t numElems,
	unsigned int zero_or_one)
{
	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (glbl_t_idx >= numElems || d_preds[glbl_t_idx] == 0)
	{
		return;
	}

	unsigned int d_out_idx = d_scanned_preds[glbl_t_idx];
	// offset the addresses with total sum of predicate 
	//  array when working with 1 bits
	if (zero_or_one == 1)
		d_out_idx = d_out_idx + *d_out_offset;
	unsigned int curr_val = d_in[glbl_t_idx];
	d_out[d_out_idx] = curr_val;
}

void radix_sort(unsigned int* const d_out,
	unsigned int* const d_in,
	unsigned int* const d_preds,
	unsigned int* const d_scanned_preds,
	const size_t numElems)
{
	unsigned int block_sz = 1024;
	unsigned int grid_sz = (unsigned int)ceil(float(numElems) / float(block_sz));

	unsigned int* d_scatter_offset;
	checkCudaErrors(cudaMalloc(&d_scatter_offset, sizeof(unsigned int)));

	// Do this for every bit, from LSB to MSB
	for (unsigned int sw = 0; sw < (sizeof(unsigned int) * 8); ++sw)
	{
		for (unsigned int bit = 0; bit <= 1; ++bit)
		{
			unsigned int bit_mask = 1 << sw;

			// Build predicate array
			gpu_build_pred<<<grid_sz, block_sz>>>(d_preds, d_in, numElems, bit_mask, bit);

			// Scan predicate array
			//  If working with 0's, make sure the total sum of the predicate 
			//  array is recorded for determining the offset of the 1's
			if (bit == 0)
				sum_scan_blelloch(d_scanned_preds, d_scatter_offset, d_preds, numElems);
			else
				sum_scan_blelloch(d_scanned_preds, NULL, d_preds, numElems);

			// Scatter d_in's elements to their new locations in d_out
			//  Use predicate array to figure out which threads will move
			//  Use scanned predicate array to figure out the locations
			gpu_scatter_elems<<<grid_sz, block_sz>>>(d_out, d_in, d_preds, d_scanned_preds, d_scatter_offset, numElems, bit);
		}

		// Copy d_out to d_in in preparation for next significant bit
		checkCudaErrors(cudaMemcpy(d_in, d_out, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	}	

	checkCudaErrors(cudaFree(d_scatter_offset));
}
