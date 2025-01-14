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

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "pargeo/unionFind.h"


namespace po = boost::program_options;


uint32_t compute_dep_ptr(float* query_ptr, float query_density, float* data, std::vector<float>& densities, const size_t data_aligned_dim, const unsigned L,
		diskann::Index<float, uint32_t>& index){
	diskann::DistanceL2Float distance_metric;
	if(L*4 > densities.size()) return densities.size();
	
	std::vector<uint32_t> query_result_id(L, densities.size());
	index.search(query_ptr, L, L, query_result_id.data());

	float minimum_dist = std::numeric_limits<float>::max();
	uint32_t dep_ptr = densities.size();
	for(unsigned i=0; i<L; i++){
		uint32_t id = query_result_id[i];
		if(id == densities.size()) break;
		if(densities[id] > query_density){
			float* ptr = data + id * data_aligned_dim;
			float dist = distance_metric.compare(query_ptr, ptr, data_aligned_dim);
			if(dist < minimum_dist){
				minimum_dist = dist;
				dep_ptr = id;
			}
		}
	}
	if(dep_ptr == densities.size()){
		return compute_dep_ptr(query_ptr, query_density, data, densities, data_aligned_dim, L*2, index);
	}
	return dep_ptr;
}

float compute_density(float* query_ptr, float* data, const size_t data_aligned_dim, const unsigned K, const unsigned L, 
		diskann::Index<float, uint32_t>& index){
	diskann::DistanceL2Float distance_metric;
	std::vector<uint32_t> query_result_id(K, 0);
	index.search(query_ptr, K, L, query_result_id.data());
	float* knn = data + query_result_id[K-1] * data_aligned_dim;
	float distance = distance_metric.compare(query_ptr, knn, data_aligned_dim);
	if(distance <= 0){
		return std::numeric_limits<float>::max();
	}else{
		return 1/distance;
	}
}


void dpc(const unsigned K, const unsigned L, const unsigned Lnn, const unsigned num_threads, const std::string& data_path, const std::string& output_path, const unsigned Lbuild=100, const unsigned max_degree=64, const float alpha=1.2){
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;


	float* data = nullptr;
	size_t data_num, data_dim, data_aligned_dim;
	diskann::load_text_file(data_path, data, data_num, data_dim,
                               data_aligned_dim);
	//diskann::load_aligned_bin<float>(data_path, data, data_num, data_dim,
      //                         data_aligned_dim);


	std::cout<<"data_num: "<<data_num<<std::endl;

	diskann::Metric metric = diskann::Metric::L2;
	diskann::Parameters paras;
	paras.Set<unsigned>("R", max_degree);
	paras.Set<unsigned>("L", Lbuild);
	paras.Set<unsigned>("Lnn", 2048); // check memory usage
	paras.Set<unsigned>("C", 750);  // maximum candidate set size during pruning procedure
	paras.Set<float>("alpha", alpha);
	paras.Set<bool>("saturate_graph", 0);
	paras.Set<unsigned>("num_threads", num_threads);

	auto pt1 = high_resolution_clock::now();

	diskann::Index<float, uint32_t> index(metric, data_dim, data_num, false, false, false,
	                            false, false, false);
	index.build(data, data_num, paras);

	auto pt2 = high_resolution_clock::now();
	std::cout<<"begin density computation"<<std::endl;

	std::vector<float> densities(data_num);

	omp_set_num_threads(num_threads);
	#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) data_num; i++) {
    	densities[i] = compute_density(data + i*data_aligned_dim, data, data_aligned_dim, K, L, index);
    }

    auto pt3 = high_resolution_clock::now();
    std::cout<<"begin dependent point computation"<<std::endl;

    std::vector<uint32_t> dep_ptrs(data_num);

    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) data_num; i++) {
    	dep_ptrs[i] = compute_dep_ptr(data + i*data_aligned_dim, densities[i], data, densities, data_aligned_dim, Lnn, index);
    }

    diskann::aligned_free(data);

    auto pt4 = high_resolution_clock::now();

    pargeo::unionFind<int> UF(densities.size());
	parlay::parallel_for(0, densities.size(), [&](int i){
		if(dep_ptrs[i] != densities.size())
			UF.link(i, dep_ptrs[i]);
	});
	std::vector<int> cluster(densities.size());
	parlay::parallel_for(0, densities.size(), [&](int i){
		cluster[i] = UF.find(i);
	});

	auto pt5 = high_resolution_clock::now();

    std::cout<<"finish all"<<std::endl;

    std::cout<<"1. index construction time \n2. density computation time \n3. dependent point computation time \n4. clustering time"<<std::endl;
    std::cout<<duration_cast<microseconds>(pt2-pt1).count()/1000000.0<<std::endl;
    std::cout<<duration_cast<microseconds>(pt3-pt2).count()/1000000.0<<std::endl;
    std::cout<<duration_cast<microseconds>(pt4-pt3).count()/1000000.0<<std::endl;
    std::cout<<duration_cast<microseconds>(pt5-pt4).count()/1000000.0<<std::endl;
    // stop here

    if(output_path != ""){
    	std::ofstream fout(output_path);
    	for (size_t i = 0; i < cluster.size(); i++){
    		fout << cluster[i] << std::endl;
    	}
    	fout.close();
	}
}




int main(int argc, char** argv){
	std::string query_file, output_file;
	po::options_description desc{"Arguments"};
 	try {
	    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
	    desc.add_options()("output_file",
                       po::value<std::string>(&output_file)->default_value(""),
                       "Output file in binary format");
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
	dpc(6, 12, 2, 8, query_file, output_file, 12, 4);
}