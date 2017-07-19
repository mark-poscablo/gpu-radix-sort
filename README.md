# GPU Radix Sort
CUDA implementation of parallel radix sort using Blelloch scan
- Implementation of 4-way radix sort as described in this [paper by Ha, Krüger, and Silva](https://vgc.poly.edu/~csilva/papers/cgf.pdf)
- 2 bits per pass, resulting in 4-way split each pass
- No order checking at every pass yet
- Each block uses a bank conflict-free Blelloch scan described in this [presentation by Mark Harris](https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf)
- Each block sorts its own local portion of the global array for greater memory coalescing during global shuffles
- Prefix summing the global block sums uses the [large-scale bank-conflict free Blelloch scan](https://github.com/mark-poscablo/gpu-prefix-sum)
- For **randomly ordered** 134 million unsigned ints, **this outperforms** `std::sort()` by about **4.22x**
- For **descendingly ordered** 134 million unsigned ints, **`std::sort()` outperforms** this by about **1.83x**
- The results above were observed using a p2.xlarge AWS instance running the NVIDIA CUDA Toolkit 7.5 AMI. The instance is equipped with 12 EC2 Compute Units (4 virtual cores), plus 1 NVIDIA K80 (GK210) GPU.
