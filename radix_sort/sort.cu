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

//__global__
//void gpu_build_pred(unsigned int* const d_out,
//	unsigned int* const d_in,
//	const size_t numElems,
//	unsigned int bit_mask,
//	unsigned int zero_or_one)
//{
//	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if (glbl_t_idx >= numElems)
//		return;
//
//	unsigned int curr_elem = d_in[glbl_t_idx];
//	// predicate is true if result is 0
//	unsigned int pred = curr_elem & bit_mask;
//	unsigned int pred_result = zero_or_one ? 0 : 1;
//	if (pred == bit_mask)
//	{
//		pred_result = zero_or_one ? 1 : 0;
//	}
//	d_out[glbl_t_idx] = pred_result;
//
//	__syncthreads();
//
//	unsigned int dummy = d_out[glbl_t_idx];
//}
//
//__global__
//void gpu_scatter_elems(unsigned int* const d_out,
//	unsigned int* const d_in,
//	unsigned int* const d_preds,
//	unsigned int* const d_scanned_preds,
//	unsigned int* const d_out_offset,
//	const size_t numElems,
//	unsigned int zero_or_one)
//{
//	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if (glbl_t_idx >= numElems || d_preds[glbl_t_idx] == 0)
//	{
//		return;
//	}
//
//	unsigned int d_out_idx = d_scanned_preds[glbl_t_idx];
//	// offset the addresses with total sum of predicate 
//	//  array when working with 1 bits
//	if (zero_or_one == 1)
//		d_out_idx = d_out_idx + *d_out_offset;
//	unsigned int curr_val = d_in[glbl_t_idx];
//	d_out[d_out_idx] = curr_val;
//}
//
//void radix_sort(unsigned int* const d_out,
//	unsigned int* const d_in,
//	unsigned int* const d_preds,
//	unsigned int* const d_scanned_preds,
//	const size_t numElems)
//{
//	unsigned int block_sz = 1024;
//	// Instead of using ceiling and risking miscalculation due to precision, just automatically  
//	//  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
//	//unsigned int grid_sz = (unsigned int)std::ceil((double)numElems / (double)block_sz);
//	unsigned int grid_sz = numElems / block_sz;
//	if (numElems % block_sz != 0)
//		grid_sz += 1;
//
//	unsigned int* d_scatter_offset;
//	checkCudaErrors(cudaMalloc(&d_scatter_offset, sizeof(unsigned int)));
//
//	// Do this for every bit, from LSB to MSB
//	for (unsigned int sw = 0; sw < (sizeof(unsigned int) * 8); ++sw)
//	{
//		for (unsigned int bit = 0; bit <= 1; ++bit)
//		{
//			unsigned int bit_mask = 1 << sw;
//
//			// Build predicate array
//			gpu_build_pred << <grid_sz, block_sz >> >(d_preds, d_in, numElems, bit_mask, bit);
//
//			// Scan predicate array
//			//  If working with 0's, make sure the total sum of the predicate 
//			//  array is recorded for determining the offset of the 1's
//			if (bit == 0)
//				sum_scan_blelloch(d_scanned_preds, d_scatter_offset, d_preds, numElems);
//			else
//				sum_scan_blelloch(d_scanned_preds, NULL, d_preds, numElems);
//
//			// Scatter d_in's elements to their new locations in d_out
//			//  Use predicate array to figure out which threads will move
//			//  Use scanned predicate array to figure out the locations
//			gpu_scatter_elems << <grid_sz, block_sz >> >(d_out, d_in, d_preds, d_scanned_preds, d_scatter_offset, numElems, bit);
//		}
//
//		// Copy d_out to d_in in preparation for next significant bit
//		checkCudaErrors(cudaMemcpy(d_in, d_out, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
//	}
//
//	checkCudaErrors(cudaFree(d_scatter_offset));
//}

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

	extern __shared__ unsigned int shmem[];
	unsigned int* s_data = shmem;
	// s_mask_out[] will be scanned in place
	unsigned int* s_mask_out = &s_data[max_elems_per_block]; 
	unsigned int* s_merged_scan_mask_out = &s_mask_out[max_elems_per_block + (max_elems_per_block >> LOG_NUM_BANKS)];
	unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
	unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];

	unsigned int thid = threadIdx.x;
	unsigned int ai = thid;
	unsigned int bi = thid + blockDim.x;

	// Copy block's portion of global input data to shared memory
	unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
	if (cpy_idx < d_in_len)
	{
		s_data[ai] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < d_in_len)
			s_data[bi] = d_in[cpy_idx + blockDim.x];
	}
	__syncthreads();

	// To extract the correct 2 bits, we first shift the number
	//  to the right until the correct 2 bits are in the 2 LSBs,
	//  then mask on the number with 11 (3) to remove the bits
	//  on the left
	unsigned int ai_2bit_extract = (s_data[ai] >> input_shift_width) & 3;
	unsigned int bi_2bit_extract = (s_data[bi] >> input_shift_width) & 3;

	for (unsigned int i = 0; i < 4; ++i)
	{
		bool ai_val_equals_i = ai_2bit_extract == i;
		bool bi_val_equals_i = bi_2bit_extract == i;

		// build bit mask output
		if (cpy_idx < d_in_len)
		{
			s_mask_out[ai + CONFLICT_FREE_OFFSET(ai)] = ai_val_equals_i;
			if (cpy_idx + blockDim.x < d_in_len)
				s_mask_out[bi + CONFLICT_FREE_OFFSET(bi)] = bi_val_equals_i;
		}

		// scan bit mask output

		// Upsweep/Reduce step
		int offset = 1;
		for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
		{
			__syncthreads();

			if (thid < d)
			{
				int ai = offset * ((thid << 1) + 1) - 1;
				int bi = offset * ((thid << 1) + 2) - 1;
				ai += CONFLICT_FREE_OFFSET(ai);
				bi += CONFLICT_FREE_OFFSET(bi);

				s_mask_out[bi] += s_mask_out[ai];
			}
			offset <<= 1;
		}

		// Save the total sum on the global block sums array
		// Then clear the last element on the shared memory
		if (thid == 0)
		{
			unsigned int total_sum = s_mask_out[max_elems_per_block - 1
				+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];;
			s_mask_out_sums[i] = total_sum;
			d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
			s_mask_out[max_elems_per_block - 1
				+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
		}

		// Downsweep step
		for (int d = 1; d < max_elems_per_block; d <<= 1)
		{
			offset >>= 1;
			__syncthreads();

			if (thid < d)
			{
				int ai = offset * ((thid << 1) + 1) - 1;
				int bi = offset * ((thid << 1) + 2) - 1;
				ai += CONFLICT_FREE_OFFSET(ai);
				bi += CONFLICT_FREE_OFFSET(bi);

				unsigned int temp = s_mask_out[ai];
				s_mask_out[ai] = s_mask_out[bi];
				s_mask_out[bi] += temp;
			}
		}
		__syncthreads();

		if (ai_val_equals_i)
		{
			s_merged_scan_mask_out[ai] = s_mask_out[ai + CONFLICT_FREE_OFFSET(ai)];
			//d_prefix_sums[cpy_idx] = s_mask_out[ai + CONFLICT_FREE_OFFSET(ai)];
		}

		if (bi_val_equals_i)
		{
			s_merged_scan_mask_out[bi] = s_mask_out[bi + CONFLICT_FREE_OFFSET(bi)];
			//d_prefix_sums[cpy_idx + blockDim.x] = s_mask_out[bi + CONFLICT_FREE_OFFSET(bi)];
		}
	}
	__syncthreads();

	// Scan mask output sums
	// Just do a naive scan since the array is really small
	if (thid == 0)
	{
		unsigned int run_sum = 0;
		for (unsigned int i = 0; i < 4; ++i)
		{
			s_scan_mask_out_sums[i] = run_sum;
			run_sum += s_mask_out_sums[i];
		}
	}
	__syncthreads();

	// Calculate the new indices of the input elements for sorting
	unsigned int meow = s_merged_scan_mask_out[ai];
	unsigned int arf = s_scan_mask_out_sums[ai_2bit_extract];
	unsigned int new_ai = meow + arf;
	//unsigned int new_ai = s_merged_scan_mask_out[ai] + s_scan_mask_out_sums[ai_2bit_extract];
	unsigned int new_bi = s_merged_scan_mask_out[bi] + s_scan_mask_out_sums[bi_2bit_extract];

	// Shuffle the block's input elements to actually sort them
	unsigned int ai_data = s_data[ai];
	unsigned int ai_prefix_sum = s_merged_scan_mask_out[ai];
	unsigned int bi_data = s_data[bi];
	unsigned int bi_prefix_sum = s_merged_scan_mask_out[bi];
	__syncthreads();
	s_data[new_ai] = ai_data;
	s_merged_scan_mask_out[new_ai] = ai_prefix_sum;
	s_data[new_bi] = bi_data;
	s_merged_scan_mask_out[new_bi] = bi_prefix_sum;
	__syncthreads();

	// copy block-wise sort results to global memory
	d_out_sorted[cpy_idx] = s_data[ai];
	d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[ai];
	if (cpy_idx + blockDim.x < d_in_len)
	{
		d_out_sorted[cpy_idx + blockDim.x] = s_data[bi];
		d_prefix_sums[cpy_idx + blockDim.x] = s_merged_scan_mask_out[bi];
	}
}

__global__ void gpu_glbl_shuffle(unsigned int* d_out,
	unsigned int* d_in,
	unsigned int* d_scan_block_sums,
	unsigned int* d_prefix_sums,
	unsigned int input_shift_width,
	unsigned int d_in_len,
	unsigned int max_elems_per_block)
{
	// get d = digit
	// get n = blockIdx
	// get m = local prefix sum array value
	// calculate global position = P_d[n] + m
	// copy input element to final position in d_out

	unsigned int thid = threadIdx.x;

	unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
	
	unsigned int ai_data = d_in[cpy_idx];
	unsigned int bi_data = d_in[cpy_idx + blockDim.x];
	
	unsigned int ai_2bit_extract = 0;
	unsigned int bi_2bit_extract = 0;
	
	if (cpy_idx < d_in_len)
	{
		ai_2bit_extract = (ai_data >> input_shift_width) & 3;
		if (cpy_idx + blockDim.x < d_in_len)
			bi_2bit_extract = (bi_data >> input_shift_width) & 3;
		else
			return;
	}
	else
	{
		return;
	}

	unsigned int ai_prefix_sum = d_prefix_sums[cpy_idx];
	unsigned int bi_prefix_sum = d_prefix_sums[cpy_idx + blockDim.x];

	unsigned int ai_glbl_pos = d_scan_block_sums[ai_2bit_extract * gridDim.x + blockIdx.x]
								+ d_prefix_sums[cpy_idx];
	unsigned int bi_glbl_pos = d_scan_block_sums[bi_2bit_extract * gridDim.x + blockIdx.x]
								+ d_prefix_sums[cpy_idx + blockDim.x];

	__syncthreads();

	d_out[ai_glbl_pos] = ai_data;
	d_out[bi_glbl_pos] = bi_data;
}

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
void radix_sort(unsigned int* const d_out,
	unsigned int* const d_in,
	unsigned int d_in_len)
{
	unsigned int block_sz = MAX_BLOCK_SZ / 2;
	unsigned int max_elems_per_block = block_sz * 2;
	unsigned int grid_sz = d_in_len / max_elems_per_block;
	// Take advantage of the fact that integer division drops the decimals
	if (d_in_len % max_elems_per_block != 0)
		grid_sz += 1;

	unsigned int* d_prefix_sums;
	unsigned int d_prefix_sums_len = d_in_len;
	checkCudaErrors(cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * d_prefix_sums_len));
	checkCudaErrors(cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_prefix_sums_len));

	unsigned int* d_block_sums;
	unsigned int d_block_sums_len = 4 * grid_sz; // 4-way split
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len));

	unsigned int* d_scan_block_sums;
	checkCudaErrors(cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len));
	checkCudaErrors(cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len));

	// shared memory consists of 3 arrays the size of the block-wise input
	//  and 2 arrays the size of n in the current n-way split (4)
	unsigned int s_data_len = max_elems_per_block;
	unsigned int s_mask_out_len = max_elems_per_block + (max_elems_per_block / NUM_BANKS);
	unsigned int s_merged_scan_mask_out_len = max_elems_per_block;
	unsigned int s_mask_out_sums_len = 4; // 4-way split
	unsigned int s_scan_mask_out_sums_len = 4;
	unsigned int shmem_sz = (s_data_len 
							+ s_mask_out_len
							+ s_merged_scan_mask_out_len
							+ s_mask_out_sums_len
							+ s_scan_mask_out_sums_len)
							* sizeof(unsigned int);


	// for every 2 bits from LSB to MSB:
	//  block-wise radix sort (write blocks back to global memory)
	for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 2)
	{
		gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_in, 
																d_prefix_sums, 
																d_block_sums, 
																shift_width, 
																d_in, 
																d_in_len, 
																max_elems_per_block);

		//unsigned int* h_test = new unsigned int[d_in_len];
		//checkCudaErrors(cudaMemcpy(h_test, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToHost));
		//for (unsigned int i = 0; i < d_in_len; ++i)
		//	std::cout << h_test[i] << " ";
		//std::cout << std::endl;
		//delete[] h_test;

		// scan global block sum array
		sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

		// scatter/shuffle block-wise sorted array to final positions
		gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in, 
													d_in, 
													d_scan_block_sums, 
													d_prefix_sums, 
													shift_width, 
													d_in_len, 
													max_elems_per_block);
	}

	checkCudaErrors(cudaFree(d_scan_block_sums));
	checkCudaErrors(cudaFree(d_block_sums));
	checkCudaErrors(cudaFree(d_prefix_sums));

	checkCudaErrors(cudaMemcpy(d_out, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice));
}
