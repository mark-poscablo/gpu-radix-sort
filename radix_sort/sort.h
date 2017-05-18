#ifndef SORT_H__
#define SORT_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.h"

void radix_sort(unsigned int* const d_out,
	unsigned int* const d_in,
	unsigned int* const d_preds,
	unsigned int* const d_scanned_preds,
	const size_t numElems);

#endif