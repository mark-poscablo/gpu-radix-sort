# GPU Radix Sort
CUDA implementation of parallel radix sort using Blelloch scan
- Implementation of 4-way radix sort as described in this [paper by Ha, Krüger, and Silva](https://vgc.poly.edu/~csilva/papers/cgf.pdf)
- 2 bits per pass, resulting in 4-way split each pass
- No order checking at every pass yet
- Each block's internal scans now use Hillis-Steele instead of Blelloch, since the internal scan's input size is roughly the same size as the number of threads per block. In this case, Hillis-Steele's larger work complexity than Blelloch's is worth having for Hillis-Steele halving the span of Blelloch's. 
- Each block sorts its own local portion of the global array for greater memory coalescing during global shuffles
- Prefix summing the global block sums uses the [large-scale bank-conflict free Blelloch scan](https://github.com/mark-poscablo/gpu-prefix-sum), which in turn uses the padded addressing solution for bank conflicts, described in this [presentation by Mark Harris](https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf)
- For **randomly ordered** 134 million unsigned ints, **this outperforms** `std::sort()` by about **9.84x**
- For **descendingly ordered** 134 million unsigned ints, **this outperforms** `std::sort()` by about **1.30x**
- The results above were observed using a p2.xlarge AWS instance running the NVIDIA CUDA Toolkit 7.5 AMI. The instance is equipped with 12 EC2 Compute Units (4 virtual cores), plus 1 NVIDIA K80 (GK210) GPU.
