#include "sort.h"

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

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
	// Instead of using ceiling and risking miscalculation due to precision, just automatically  
	//  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
	//unsigned int grid_sz = (unsigned int)std::ceil((double)numElems / (double)block_sz);
	unsigned int grid_sz = numElems / block_sz;
	if (numElems % block_sz != 0)
		grid_sz += 1;

	unsigned int* d_scatter_offset;
	checkCudaErrors(cudaMalloc(&d_scatter_offset, sizeof(unsigned int)));

	// Do this for every bit, from LSB to MSB
	for (unsigned int sw = 0; sw < (sizeof(unsigned int) * 8); ++sw)
	{
		for (unsigned int bit = 0; bit <= 1; ++bit)
		{
			unsigned int bit_mask = 1 << sw;

			// Build predicate array
			gpu_build_pred << <grid_sz, block_sz >> >(d_preds, d_in, numElems, bit_mask, bit);

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
			gpu_scatter_elems << <grid_sz, block_sz >> >(d_out, d_in, d_preds, d_scanned_preds, d_scatter_offset, numElems, bit);
		}

		// Copy d_out to d_in in preparation for next significant bit
		checkCudaErrors(cudaMemcpy(d_in, d_out, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	}

	checkCudaErrors(cudaFree(d_scatter_offset));
}

__global__ void gpu_radix_sort_local(unsigned int* d_out_sorted,
	unsigned int* d_prefix_sums,
	unsigned int* d_block_sums,
	unsigned int input_shift_width,
	unsigned int* d_in,
	unsigned int d_in_len,
	unsigned int max_elems_per_block)
{
	// need shared memory array for:
	// - block's share of the input data (local sort will be put here too)
	// - mask outputs
	// - scanned mask outputs
	// - merged scaned mask outputs ("local prefix sum")
	// - local sums of scanned mask outputs
	// - scanned local sums of scanned mask outputs

	// for all radix combinations:
	//  build mask output for current radix combination
	//  scan mask ouput
	//  store needed value from current prefix sum array to merged prefix sum array
	//  store total sum of mask output (obtained from scan) to global block sum array
	// calculate local sorted address from local prefix sum and scanned mask output's total sums
	// shuffle input block according to calculated local sorted addresses
	// shuffle local prefix sums according to calculated local sorted addresses
	// copy locally sorted array back to global memory
	// copy local prefix sum array back to global memory
}

__global__ void gpu_glbl_shuffle()
{
	// get d = digit
	// get n = blockIdx
	// get m = local prefix sum array value
	// calculate global position P_d[n] + m
	// copy input element to final position in d_out
}

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
void radix_sort_4way()
{
	// for every 2 bits from LSB to MSB:
	//  block-wise radix sort (write blocks back to global memory)

	//  scan global block sum array

	//  scatter/shuffle block-wise sorted array to final positions

	//  copy d_out to d_in in prep for next pass
}
