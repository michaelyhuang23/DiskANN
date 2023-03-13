#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "index.h"
#include "distance.h"
#include "memory_mapper.h"
#include "utils.h"
#include "ann_exception.h"

namespace po = boost::program_options;


uint32_t compute_dep_ptr(float* query_ptr, float query_density, float* data, std::vector<float>& densities, const size_t data_aligned_dim, const unsigned L,
		diskann::Index<float, uint32_t>& index, diskann::Distance<float>* distance_metric){
	std::vector<uint32_t> query_result_id(L, 0);
	index.search(query_ptr, L, L, query_result_id.data());

	float minimum_dist = std::numeric_limits<float>::max();
	uint32_t dep_ptr = 0;
	for(unsigned i=0; i<L; i++){
		uint32_t id = query_result_id[i];
		if(densities[id] > query_density){
			float* ptr = data + id * data_aligned_dim;
			float dist = distance_metric->compare(query_ptr, ptr, data_aligned_dim);
			if(dist < minimum_dist){
				minimum_dist = dist;
				dep_ptr = id;
			}
		}
	}

	return dep_ptr;
}

float compute_density(float* query_ptr, float* data, const size_t data_aligned_dim, const unsigned K, const unsigned L, 
		diskann::Index<float, uint32_t>& index, diskann::Distance<float>* distance_metric){
	std::vector<uint32_t> query_result_id(K, 0);
	index.search(query_ptr, K, L, query_result_id.data());
	float* knn = data + query_result_id[K-1] * data_aligned_dim;
	float distance = distance_metric->compare(query_ptr, knn, data_aligned_dim);
	if(distance <= 0){
		return std::numeric_limits<float>::max();
	}else{
		return 1/distance;
	}
}

void dpc(const unsigned K, const unsigned L, const unsigned num_threads, const std::string& data_path, const unsigned Lbuild=100, const unsigned max_degree=64, const float alpha=1.2){
	float* data = nullptr;
	size_t data_num, data_dim, data_aligned_dim;
	diskann::load_aligned_bin<float>(data_path, data, data_num, data_dim,
                               data_aligned_dim);

	diskann::Metric metric = diskann::Metric::L2;
	diskann::Distance<float>* distance_metric = diskann::get_distance_function<float>(metric);

	diskann::Parameters paras;
	paras.Set<unsigned>("R", max_degree);
	paras.Set<unsigned>("L", Lbuild);
	paras.Set<unsigned>("C", 750);  // maximum candidate set size during pruning procedure
	paras.Set<float>("alpha", alpha);
	paras.Set<bool>("saturate_graph", 0);
	paras.Set<unsigned>("num_threads", num_threads);

	diskann::Index<float, uint32_t> index(metric, data_dim, data_num, false, false, false,
	                            false, false, false);
	index.build(data_path.c_str(), data_num, paras);

	std::vector<float> densities(data_num);

	#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) data_num; i++) {
    	densities[i] = compute_density(data + i*data_aligned_dim, data, data_aligned_dim, K, L, index, distance_metric);
    }

    std::vector<uint32_t> dep_ptrs(data_num);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) data_num; i++) {
    	dep_ptrs[i] = compute_dep_ptr(data + i*data_aligned_dim, densities[i], data, densities, data_aligned_dim, L, index, distance_metric);
    }
    // stop here
}



int main(int argc, char** argv){
	std::string query_file;
	po::options_description desc{"Arguments"};
 	try {
	    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
	    po::variables_map vm;
	    po::store(po::parse_command_line(argc, argv, desc), vm);
    	if (vm.count("help")) {
	    	std::cout << desc;
    		return 0;
	    }
    	po::notify(vm);	
	}catch(const std::exception& ex) {
    	std::cerr << ex.what() << '\n';
	    return -1;
	}
	dpc(6, 12, 6, query_file);
}