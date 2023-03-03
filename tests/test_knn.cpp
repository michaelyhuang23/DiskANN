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
#include "memory_mapper.h"
#include "utils.h"
#include "ann_exception.h"

namespace po = boost::program_options;


void queryKNN(const unsigned K, const unsigned L, const unsigned num_threads, const std::string& data_path, const unsigned Lbuild=100, const unsigned max_degree=64, const float alpha=1.2){
	float* query = nullptr;
	size_t query_num, query_dim, query_aligned_dim;
	diskann::load_aligned_bin<float>(data_path, query, query_num, query_dim,
                               query_aligned_dim);

	diskann::Metric metric = diskann::Metric::L2;

	diskann::Parameters paras;
	paras.Set<unsigned>("R", max_degree);
	paras.Set<unsigned>("L", Lbuild);
	paras.Set<unsigned>("C", 750);  // maximum candidate set size during pruning procedure
	paras.Set<float>("alpha", alpha);
	paras.Set<bool>("saturate_graph", 0);
	paras.Set<unsigned>("num_threads", num_threads);

	uint64_t data_num, data_dim;
	diskann::get_bin_metadata(data_path, data_num, data_dim);
	diskann::Index<float, uint32_t> index(metric, data_dim, data_num, false, false, false,
	                            false, false, false);
	index.build(data_path.c_str(), data_num, paras);

	
    if (L < K) {
    	diskann::cout << "L smaller than K" << std::endl; 
    	return;
    }

    std::vector<uint32_t> query_result_id(K, 0);
    index.search(query, K, L, query_result_id.data());

    for(size_t i=0; i<K; ++i)
	    std::cout << query_result_id[i] <<std::endl;
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
	queryKNN(2, 4, 6, query_file);
}