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


std::pair<uint32_t, float> compute_dep_ptr(float* query_ptr, float query_density, float* data, std::vector<float>& densities, const size_t data_aligned_dim, const unsigned L,
		diskann::Index<float, uint32_t>& index){
	if(L*4 > densities.size()) return std::make_pair(densities.size(), std::numeric_limits<float>::max());
	
	std::vector<uint32_t> query_result_id(1, densities.size());
	std::vector<float> distances(1, std::numeric_limits<float>::max());
	index.search_density(query_ptr, (size_t)1, L, query_density, (unsigned)(densities.size()-1), densities, query_result_id.data(), distances.data());

	return std::make_pair(query_result_id[0], distances[0]);
}

float compute_density(float* query_ptr, float* data, const size_t data_aligned_dim, const unsigned K, const unsigned L, 
		diskann::Index<float, uint32_t>& index){
	diskann::DistanceL2Float distance_metric;
	std::vector<uint32_t> query_result_id(K, 0);
	index.search(query_ptr, K, L, query_result_id.data());
	float* knn = data + query_result_id[K-1] * data_aligned_dim;
	float distance = sqrt(abs(distance_metric.compare(query_ptr, knn, data_aligned_dim)));
	if(distance <= 0){
		return std::numeric_limits<float>::max();
	}else{
		return 1/distance;
	}
} // store knn

void dpc(const unsigned K, const unsigned L, const unsigned Lnn, const unsigned num_threads, const float density_cutoff, const float dist_cutoff, const std::string& data_path, const std::string& output_path, const std::string& decision_graph_path, const unsigned Lbuild=100, const unsigned max_degree=64, const float alpha=1.2){
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;

    diskann::cout<<__cplusplus<<std::endl;


	float* data = nullptr;
	size_t data_num, data_dim, data_aligned_dim;
	//diskann::load_aligned_bin<float>(data_path, data, data_num, data_dim,
      //                         data_aligned_dim);
	diskann::load_text_file(data_path, data, data_num, data_dim,
                               data_aligned_dim);

	diskann::cout<<"data_num: "<<data_num<<std::endl;

	diskann::Metric metric = diskann::Metric::L2;
	
	diskann::Parameters paras;
	paras.Set<unsigned>("R", max_degree);
	paras.Set<unsigned>("L", Lbuild);
	paras.Set<unsigned>("C", 750);  // maximum candidate set size during pruning procedure
	paras.Set<float>("alpha", alpha);
	paras.Set<unsigned>("Lnn", Lnn);
	paras.Set<bool>("saturate_graph", 0);
	paras.Set<unsigned>("num_threads", num_threads);

	auto pt1 = high_resolution_clock::now();

	diskann::Index<float, uint32_t> index(metric, data_dim, data_num, false, false, false,
	                            false, false, false);
	index.build(data, data_num, paras);

	auto pt2 = high_resolution_clock::now();
	diskann::cout<<"begin density computation"<<std::endl;

	std::vector<float> densities(data_num);

	#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) data_num; i++) {
    	densities[i] = compute_density(data + i*data_aligned_dim, data, data_aligned_dim, K, L, index);
    }

    auto pt3 = high_resolution_clock::now();
    diskann::cout<<"begin dependent point computation"<<std::endl;

    std::vector<uint32_t> dep_ptrs(data_num);
    std::vector<float> dep_dist(data_num);

    diskann::DistanceL2Float distance_metric;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) data_num; i++) {
    	auto retval = compute_dep_ptr(data + i*data_aligned_dim, densities[i], data, densities, data_aligned_dim, Lnn, index);
    	dep_ptrs[i] = retval.first;
    	dep_dist[i] = retval.second;
    }

    auto pt4 = high_resolution_clock::now();

    if(decision_graph_path != ""){    	
    	std::ofstream fout(decision_graph_path);
    	for (size_t i = 0; i < data_num; i++){
    		fout << densities[i] << " " << dep_dist[i] << '\n';
    	}
    }

    diskann::aligned_free(data);

   	auto pt5 = high_resolution_clock::now();
   	diskann::cout<<"begin clustering"<<std::endl;

    pargeo::unionFind<int> UF(data_num);
	parlay::parallel_for(0, data_num, [&](int i){
		if(dep_ptrs[i] != data_num && (densities[i]<=density_cutoff || dep_dist[i]<=dist_cutoff)){
			UF.link(i, dep_ptrs[i]);
		}
	});
	std::vector<int> cluster(data_num);
	parlay::parallel_for(0, data_num, [&](int i){
		cluster[i] = UF.find(i);
	});

	auto pt6 = high_resolution_clock::now();

    diskann::cout<<"finish all"<<std::endl;

    diskann::cout<<"1. index construction time \n2. density computation time \n3. dependent point computation time \n4. storage time \n5. clustering time"<<std::endl;
    diskann::cout<<duration_cast<microseconds>(pt2-pt1).count()/1000000.0<<std::endl;
    diskann::cout<<duration_cast<microseconds>(pt3-pt2).count()/1000000.0<<std::endl;
    diskann::cout<<duration_cast<microseconds>(pt4-pt3).count()/1000000.0<<std::endl;
    diskann::cout<<duration_cast<microseconds>(pt5-pt4).count()/1000000.0<<std::endl;
    diskann::cout<<duration_cast<microseconds>(pt6-pt5).count()/1000000.0<<std::endl;
    // stop here

    if(output_path != ""){
    	std::ofstream fout(output_path);
    	for (size_t i = 0; i < cluster.size(); i++){
    		fout << cluster[i] << '\n';
    	}
    	fout.close();
	}
}



int main(int argc, char** argv){
	std::string query_file, output_file, decision_graph_file;
	float density_cutoff, dist_cutoff;
	po::options_description desc{"Arguments"};
 	try {
	    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
	    desc.add_options()("output_file",
                       po::value<std::string>(&output_file)->default_value(""),
                       "Output file in binary format");
	    desc.add_options()("decision_graph_file",
                       po::value<std::string>(&decision_graph_file)->default_value(""),
                       "Decision graph file in binary format");
	    desc.add_options()("density_cutoff",
				       po::value<float>(&density_cutoff)->default_value(0.0f),
				       "Density below which points are treated as noise");
	    desc.add_options()("dist_cutoff",
				       po::value<float>(&dist_cutoff)->default_value(0.0f),
				       "Density below which points are sorted into the same cluster");
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
	dpc(2, 32, 32, 4, density_cutoff, dist_cutoff, query_file, output_file, decision_graph_file, 32, 28);
}