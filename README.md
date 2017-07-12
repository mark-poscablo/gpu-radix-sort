# GPU Radix Sort
CUDA implementation of parallel radix sort using Blelloch scan
- Implementation of 4-way radix sort as described in this [paper by Ha, Krüger, and Silva](https://vgc.poly.edu/~csilva/papers/cgf.pdf)
- 2 bits per pass, resulting in 4-way split each significant bit
- No order checking at every pass yet
- Each block uses a bank conflict-free Blelloch scan described in this [presentation by Mark Harris](https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf)
- Each block sorts its own local portion of the global array for greater memory coalescing during global shuffles
- Prefix summing the global block sums uses the [large-scale bank-conflict free Blelloch scan](https://github.com/mark-poscablo/gpu-prefix-sum)
- Outperforms the previous 2-way implementation by about 1.8x, but is still outperformed by the C++ standard library sort by about 3x. These results were seen on a machine with an Intel i7-4712HQ processor and a Nvidia GeForce GTX 850M GPU.
