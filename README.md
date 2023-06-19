# DiskANN for DPC

### Build:

```bash
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 
```

### Run:

```bash
cd build/tests
./blindprobe_dpc --K 50 --L 64 --Lnn 64 --max_degree 64 --Lbuild 64 --density_cutoff 0.001 --dist_cutoff 1300 --Dbrute 1000 --query_file ../../data/mnist.data --output_file ../../results/mnist_blindprobe.out --decision_graph_file ../../results/mnist_blindprobe.graph
```

`--K` is the number of k nearest neighbors used for density computation

`--L` is the L parameter for density computation (should be larger than `K`)

`--Lnn` is the L parameter for dependent point finding

`--max_degree` is the max_degree of the graph

`--Lbuild` is used for graph construction

`--density_cutoff` decides which points are noise points

`--dist_cutoff` decides which non-noise points are cluster centers

`--Dbrute` decides how many high density points to bruteforce dependent points

`--query_file` is the txt source file. It should contain $n$ rows, each with $dim$ floats

`--output_file` stores the clustering results

`--decision_graph_file` stores the density and dependent point results

### Files:

`/tests/blindprobe_dpc.cpp` is the main dpc algorithm. It uses a priority search queue to go through points close to the query point to find the dependent point. The density it computes is an average result of several nearest neighbors, which helps stabilize the result. 

`/tests/doubling_dpc.cpp` is an archaic dpc algorithm. It doubles search range each time, until the dependent point is found. There's a parallelism bug in the diskann library which I believe is related to resizing the `scratch`. This algorithm triggers that bug.

`/tests/bruteforce_dpc.cpp` is just the bruteforce dpc algorithm. It computes a simpler version of density compared to `blindprobe_dpc`. It takes $O(n^2)$ time and linear memory. 

`/python/plot_decision_graph.py` is a helper program that plots decision graph based on the outputed `decision_graph_file`

`python/cluster_eval.py` is a helper program that computes various clustering metrics based on ground truth and the output clustering file `output_file` 

**tip: using `diskann::cout` can cause parallelism error, so using `std::cout` may be better**

### Just nearest neighbor search usage:

Please see the following pages on using the compiled code:

- [Commandline interface for building and search SSD based indices](workflows/SSD_index.md)  
- [Commandline interface for building and search in memory indices](workflows/in_memory_index.md) 
- [Commandline examples for using in-memory streaming indices](workflows/dynamic_index.md)
