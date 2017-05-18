#include "scan.h"

#define MAX_BLOCK_SZ 1024

__global__
void gpu_sum_scan_naive(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	// Using naive scan where each thread calculates a separate partial sum
	// Step complexity is still O(n) as the last thread will calculate the global sum

	unsigned int d_hist_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (d_hist_idx == 0 || d_hist_idx >= numElems)
	{
		return;
	}
	unsigned int cdf_val = 0;
	for (int i = 0; i < d_hist_idx; ++i)
	{
		cdf_val = cdf_val + d_in[i];
	}
	d_out[d_hist_idx] = cdf_val;
}

__global__
void gpu_sum_scan_blelloch(unsigned int* const d_out,
	const unsigned int* const d_in,
	unsigned int* const d_block_sums,
	const size_t numElems)
{
	extern __shared__ unsigned int s_out[];

	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	s_out[2 * threadIdx.x] = 0;
	s_out[2 * threadIdx.x + 1] = 0;

	__syncthreads();

	// Copy d_in to shared memory per block
	if (2 * glbl_t_idx < numElems)
	{
		s_out[2 * threadIdx.x] = d_in[2 * glbl_t_idx];
		if (2 * glbl_t_idx + 1 < numElems)
			s_out[2 * threadIdx.x + 1] = d_in[2 * glbl_t_idx + 1];
	}

	__syncthreads();

	// Reduce/Upsweep step

	// 2^11 = 2048, the max amount of data a block can blelloch scan
	unsigned int max_steps = 11; 

	unsigned int r_idx = 0;
	unsigned int l_idx = 0;
	unsigned int sum = 0; // global sum can be passed to host if needed
	unsigned int t_active = 0;
	for (int s = 0; s < max_steps; ++s)
	{
		t_active = 0;

		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < 2048)
			t_active = 1;

		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the actual add operation
			sum = s_out[l_idx] + s_out[r_idx];
		}
		__syncthreads();

		if (t_active)
			s_out[r_idx] = sum;
		__syncthreads();
	}

	// Copy last element (total sum of block) to block sums array
	// Then, reset last element to operation's identity (sum, 0)
	if (threadIdx.x == 0)
	{
		d_block_sums[blockIdx.x] = s_out[r_idx];
		s_out[r_idx] = 0;
	}

	__syncthreads();

	// Downsweep step

	for (int s = max_steps - 1; s >= 0; --s)
	{
		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < 2048)
		{
			t_active = 1;
		}

		unsigned int r_cpy = 0;
		unsigned int lr_sum = 0;
		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the downsweep operation
			r_cpy = s_out[r_idx];
			lr_sum = s_out[l_idx] + s_out[r_idx];
		}
		__syncthreads();

		if (t_active)
		{
			s_out[l_idx] = r_cpy;
			s_out[r_idx] = lr_sum;
		}
		__syncthreads();
	}

	// Copy the results to global memory
	if (2 * glbl_t_idx < numElems)
	{
		d_out[2 * glbl_t_idx] = s_out[2 * threadIdx.x];
		if (2 * glbl_t_idx + 1 < numElems)
			d_out[2 * glbl_t_idx + 1] = s_out[2 * threadIdx.x + 1];
	}
}

__global__
void gpu_add_block_sums(unsigned int* const d_out,
	const unsigned int* const d_in,
	unsigned int* const d_block_sums,
	const size_t numElems)
{
	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int d_block_sum_val = d_block_sums[blockIdx.x];

	unsigned int d_in_val_0 = 0;
	unsigned int d_in_val_1 = 0;

	if (2 * glbl_t_idx < numElems)
	{
		d_in_val_0 = d_in[2 * glbl_t_idx];
		if (2 * glbl_t_idx + 1 < numElems)
			d_in_val_1 = d_in[2 * glbl_t_idx + 1];
	}
	else
		return;
	__syncthreads();

	d_out[2 * glbl_t_idx] = d_in_val_0 + d_block_sum_val;
	d_out[2 * glbl_t_idx + 1] = d_in_val_1 + d_block_sum_val;
}

void sum_scan_naive(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	unsigned int blockSz = MAX_BLOCK_SZ;
	unsigned int gridSz = (unsigned int)ceil(float(numElems) / float(MAX_BLOCK_SZ));
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));
	gpu_sum_scan_naive << <gridSz, blockSz >> >(d_out, d_in, numElems);
}

void sum_scan_blelloch(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	// Zero out d_out
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));

	// Set up number of threads and blocks
	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the least number of 2048-blocks greater than the input size
	unsigned int blockSz = MAX_BLOCK_SZ;
	unsigned int max_elems_per_block = blockSz * 2; // due to binary tree nature of algorithm
	unsigned int gridSz = (unsigned int)ceil(float(numElems) / float(max_elems_per_block));

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks / grid size
	unsigned int* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * gridSz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * gridSz));

	// Sum scan data allocated for each block
	gpu_sum_scan_blelloch<<<gridSz, blockSz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);

	// Sum scan each block's total sums
	// Use basic implementation if number of total sums is <= 2048
	// Else use the more advanced implementation
	if (gridSz <= 2048)
	{
		unsigned int* d_dummy_blocks_sums;
		checkCudaErrors(cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int)));
		gpu_sum_scan_blelloch<<<1, blockSz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, gridSz);
		checkCudaErrors(cudaFree(d_dummy_blocks_sums));
	}
	else
	{
		unsigned int* d_in_block_sums;
		checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * gridSz));
		checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * gridSz, cudaMemcpyDeviceToDevice));
		sum_scan_blelloch(d_block_sums, d_in_block_sums, gridSz);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}
	
	// Uncomment to examine block sums
	//unsigned int* h_block_sums = new unsigned int[gridSz];
	//checkCudaErrors(cudaMemcpy(h_block_sums, d_block_sums, sizeof(unsigned int) * gridSz, cudaMemcpyDeviceToHost));
	//std::cout << "Block sums: ";
	//for (int i = 0; i < gridSz; ++i)
	//{
	//	std::cout << h_block_sums[i] << ", ";
	//}
	//std::cout << std::endl;
	//std::cout << "Block sums length: " << gridSz << std::endl;
	//delete[] h_block_sums;

	// Add each block's total sum to scanned output of each block
	gpu_add_block_sums<<<gridSz, blockSz>>>(d_out, d_out, d_block_sums, numElems);

	checkCudaErrors(cudaFree(d_block_sums));
}
