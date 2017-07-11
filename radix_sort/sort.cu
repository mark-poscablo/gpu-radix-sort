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
	unsigned int s_mask_out_len = max_elems_per_block + (max_elems_per_block >> LOG_NUM_BANKS);
	unsigned int* s_mask_out = &s_data[max_elems_per_block];
	unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
	unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
	unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];

	unsigned int thid = threadIdx.x;

	// Copy block's portion of global input data to shared memory
	unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
	if (cpy_idx < d_in_len)
		s_data[thid] = d_in[cpy_idx];
	else
		s_data[thid] = 0;

	__syncthreads();

	// To extract the correct 2 bits, we first shift the number
	//  to the right until the correct 2 bits are in the 2 LSBs,
	//  then mask on the number with 11 (3) to remove the bits
	//  on the left
	unsigned int t_data = s_data[thid];
	unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;

	for (unsigned int i = 0; i < 4; ++i)
	{
		// Zero out s_mask_out
		s_mask_out[thid] = 0;
		// If CONFLICT_FREE_OFFSET is used, shared memory
		//  must be a few more than 2 * blockDim.x
		if (thid + max_elems_per_block < s_mask_out_len)
			s_mask_out[thid + max_elems_per_block] = 0;
		__syncthreads();

		// build bit mask output
		bool val_equals_i = false;
		if (cpy_idx < d_in_len)
		{
			val_equals_i = t_2bit_extract == i;
			s_mask_out[thid + CONFLICT_FREE_OFFSET(thid)] = val_equals_i;
		}
		__syncthreads();

		// scan bit mask output
		// Upsweep/Reduce step
		bool t_active = thid < (blockDim.x / 2);
		int offset = 1;
		for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
		{
			__syncthreads();

			if (t_active && (thid < d))
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
			//unsigned int total_sum_idx = (unsigned int) fmin();
			unsigned int total_sum = s_mask_out[max_elems_per_block - 1
				+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
			s_mask_out_sums[i] = total_sum;
			d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
			s_mask_out[max_elems_per_block - 1
				+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
		}
		__syncthreads();

		// Downsweep step
		for (int d = 1; d < max_elems_per_block; d <<= 1)
		{
			offset >>= 1;
			__syncthreads();

			if (t_active && (thid < d))
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

		if (val_equals_i && (cpy_idx < d_in_len))
		{
			s_merged_scan_mask_out[thid] = s_mask_out[thid + CONFLICT_FREE_OFFSET(thid)];
		}
		__syncthreads();
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

	if (cpy_idx < d_in_len)
	{
		// Calculate the new indices of the input elements for sorting
		unsigned int new_pos = s_merged_scan_mask_out[thid] + s_scan_mask_out_sums[t_2bit_extract];
		//if (new_ai >= 1024)
		//	new_ai = 0;
		unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];
		
		__syncthreads();

		// Shuffle the block's input elements to actually sort them
		s_data[new_pos] = t_data;
		s_merged_scan_mask_out[new_pos] = t_prefix_sum;
		
		__syncthreads();

		// copy block-wise sort results to global 
		// then copy block-wise prefix sum results to global memory
		d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];
		d_out_sorted[cpy_idx] = s_data[thid];
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

	if (cpy_idx < d_in_len)
	{
		unsigned int t_data = d_in[cpy_idx];
		unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
		unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];
		unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x]
			+ t_prefix_sum;
		__syncthreads();
		d_out[data_glbl_pos] = t_data;
	}
}

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
void radix_sort(unsigned int* const d_out,
	unsigned int* const d_in,
	unsigned int d_in_len)
{
	unsigned int block_sz = MAX_BLOCK_SZ;
	unsigned int max_elems_per_block = block_sz;
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
		gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_out, 
													d_in, 
													d_scan_block_sums, 
													d_prefix_sums, 
													shift_width, 
													d_in_len, 
													max_elems_per_block);

		checkCudaErrors(cudaMemcpy(d_in, d_out, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice));
	}

	checkCudaErrors(cudaFree(d_scan_block_sums));
	checkCudaErrors(cudaFree(d_block_sums));
	checkCudaErrors(cudaFree(d_prefix_sums));
}
