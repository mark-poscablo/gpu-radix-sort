#include "sort.h"

__global__
void gpu_build_pred(unsigned int* const d_in,
	unsigned int* const d_preds,
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
	for (unsigned int sw = 0; sw < (sizeof(unsigned int) * 8); ++sw)
	{
		for (unsigned int bit = 0; bit <= 1; ++bit)
		{
			unsigned int bit_mask = 1 << sw;

			// Build predicate array
			gpu_build_pred << <grid_sz, block_sz >> >(d_in, d_preds, numElems, bit_mask, bit);

			// Scan predicate array
			//  If working with 0's, make sure the total sum + 1 is 
			//  recorded for the 1's
			sum_scan_blelloch(d_scanned_preds, d_preds, numElems);

			// Scatter d_in's elements to their new locations in d_out
			//  Use predicate array to figure out which threads will move
			//  Use scanned predicate array to figure out the locations
		}
	}	
}
