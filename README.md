# GPU Radix Sort
CUDA implementation of parallel radix sort using Blelloch scan
- 1 significant bit per pass, resulting in 2-way split each significant bit
- Faster than the C++ standard library sort for large enough input sizes (on my laptop, starting at 2^16 elements)
