# GPU Radix Sort
CUDA implementation of parallel radix sort using Blelloch scan
- 1 significant bit per pass, resulting in 2-way split each significant bit
- ~~Faster than the C++ standard library sort for large enough input sizes (on my laptop, starting at 2^16 elements).~~ It was only faster on the Debug configuration. In Release, the C++ standard library sort outperforms this GPU implementation for all input sizes tested so far. Most notably, the CPU implementation performs 6.7x faster than the GPU at about 134 million elements. These results were seen on a machine with an Intel i7-4712HQ processor and a Nvidia GeForce GTX 850M GPU.
