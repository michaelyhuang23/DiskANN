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


std::pair<uint32_t, float> compute_dep_ptr(float* query_ptr, uint32_t query_id, float* data, std::vector<float>& densities, const size_t data_aligned_dim, const unsigned L,
		diskann::Index<float, uint32_t>& index){
	
	std::vector<uint32_t> query_result_id(1, densities.size());
	std::vector<float> distances(1, std::numeric_limits<float>::max());
	index.search_density(query_ptr, query_id, (size_t)1, L, (unsigned)(densities.size()-1), densities, query_result_id.data(), distances.data());
	// this is a method I made that basically searches points and tracks L closest points with higher density. 
	// terminates when every thing on the to-be-visited priority queue is farther than the Lth closest point with higher density
	return std::make_pair(query_result_id[0], sqrt(abs(distances[0])));
}

std::pair<uint32_t, float> brute_compute_dep_ptr(float* query_ptr, uint32_t query_id, float* data, std::vector<float>& densities, const size_t data_aligned_dim){
	diskann::DistanceL2Float distance_metric;
	parlay::sequence<float> dists(densities.size(), std::numeric_limits<float>::max());
	parlay::parallel_for(0, densities.size(), [&](size_t j){
		if(densities[j] > densities[query_id] || (densities[j] == densities[query_id] && j < query_id)){
			dists[j] = sqrt(abs(distance_metric.compare(query_ptr, data+j*data_aligned_dim, data_aligned_dim)));
		}
	});
	auto min_it = parlay::min_element(dists);
	if(min_it == dists.end() || *min_it==std::numeric_limits<float>::max()) return std::make_pair(densities.size(), std::numeric_limits<float>::max());
	return std::make_pair(std::distance(dists.begin(), min_it), *min_it);
}

float compute_density(float* query_ptr, float* data, const size_t data_aligned_dim, const size_t data_dim, const unsigned K, const unsigned L, 
		diskann::Index<float, uint32_t>& index){
	diskann::DistanceL2Float distance_metric;
	std::vector<uint32_t> query_result_id(K, 0);
	index.search(query_ptr, K, L, query_result_id.data()); // knn result is stored into query_result_id
	float total_density = 0;
	for(uint32_t k=1;k<K;k++){
		float* knn = data + query_result_id[k] * data_aligned_dim; // gets the kth nearest neighbor
		float distance = sqrt(abs(distance_metric.compare(query_ptr, knn, data_aligned_dim))); // distance_metric gives square of distance
		float density = pow(k,1.0/data_dim)/distance; // calculates density (per length. We expect the volume density to be proportional to k/(distance^dim), so the density per length is the dim^th root of that)
		total_density += density;
	}
	return total_density / (K-1); // density estimated by averaging over kth nearest neighbors for k in range [1, K-1]. 
} 

void dpc(const unsigned K, const unsigned L, const unsigned Lnn, const unsigned num_threads, const float density_cutoff, const float dist_cutoff, const unsigned Dbrute, const std::string& data_path, const std::string& output_path, const std::string& decision_graph_path, const unsigned Lbuild=100, const unsigned max_degree=64, const float alpha=1.2){
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
	// load_text_file is a method I made for loading from txt files that contain the dataset of points. 
	// Each point occupies one row in the text file
	// load_aligned_bin loads it from a binary file of a certain format 

	diskann::cout<<"data_num: "<<data_num<<std::endl; // print number of points

	diskann::Metric metric = diskann::Metric::L2;
	
	diskann::Parameters paras;
	paras.Set<unsigned>("R", max_degree); // max_degree of the graph built
	paras.Set<unsigned>("L", Lbuild); 
	paras.Set<unsigned>("C", 750);  // maximum candidate set size during pruning procedure
	paras.Set<float>("alpha", alpha); // defined in the paper, decides edge pruning cutoff in graph construction 
	paras.Set<unsigned>("Lnn", Lnn); // L parameter used for greedy graph pruning
	paras.Set<bool>("saturate_graph", 0);
	paras.Set<unsigned>("num_threads", num_threads);

	auto pt1 = high_resolution_clock::now();

	diskann::Index<float, uint32_t> index(metric, data_dim, data_num, false, false, false,
	                            false, false, false);
	index.build(data, data_num, paras); // builds graph

	auto pt2 = high_resolution_clock::now();
	diskann::cout<<"begin density computation"<<std::endl;

	std::vector<float> densities(data_num);

	#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) data_num; i++) {
    	densities[i] = compute_density(data + i*data_aligned_dim, data, data_aligned_dim, data_dim, K, L, index);
    } // compute density

    float bruteforce_density_thres = std::numeric_limits<float>::max();
    if(Dbrute>0){
    	std::vector<float> densities_copy(densities.begin(), densities.end());
    	std::nth_element(densities_copy.begin(), densities_copy.end()-Dbrute, densities_copy.end());
    	bruteforce_density_thres = densities_copy[densities_copy.size()-Dbrute];
	} // find the density cut-off for brute force computation using nth_element

    auto pt3 = high_resolution_clock::now();
    diskann::cout<<"begin dependent point computation"<<std::endl;

    std::vector<uint32_t> dep_ptrs(data_num);
    std::vector<float> dep_dist(data_num);

    diskann::DistanceL2Float distance_metric;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) data_num; i++) {
    	if(densities[i] < bruteforce_density_thres){
    		auto retval = compute_dep_ptr(data + i*data_aligned_dim, i, data, densities, data_aligned_dim, Lnn, index);
    		dep_ptrs[i] = retval.first;    		
	    	dep_dist[i] = retval.second;	
    	}
    } // normal density computation

    parlay::parallel_for(0, data_num, [&](size_t i){
    	if(densities[i] >= bruteforce_density_thres){
    		auto retval = brute_compute_dep_ptr(data + i*data_aligned_dim, i, data, densities, data_aligned_dim);
    		dep_ptrs[i] = retval.first;
	    	dep_dist[i] = retval.second;	
    	}
    }); // bruteforce density computation. I separated the loops because parlay doesn't work well with #pragma thingy. I used parlay inside brute_compute_dep_ptr

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

    /*pargeo::unionFind<int> UF(data_num);
	parlay::parallel_for(0, data_num, [&](int i){
		if(dep_ptrs[i] != data_num && (densities[i]<=density_cutoff || dep_dist[i]<=dist_cutoff)){
			if(dep_ptrs[i]<0 || dep_ptrs[i]>data_num)
				std::cout<<i<<" "<<dep_ptrs[i]<<std::endl;
			UF.link(i, dep_ptrs[i]);
		}
	});

	std::cout<<"done merging"<<std::endl;
	std::vector<int> cluster(data_num);
	parlay::parallel_for(0, data_num, [&](int i){
		cluster[i] = UF.find(i);
	});*/

	std::vector<int> roots(data_num);
	parlay::parallel_for(0, data_num, [&](int i){
		int orig_i = i;
		while(dep_ptrs[i] != data_num && (densities[i]<=density_cutoff || dep_dist[i]<=dist_cutoff)){
			i = dep_ptrs[i];
		}
		roots[orig_i] = i; // this is for debugging only
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
    	for (size_t i = 0; i < roots.size(); i++){
    		fout << roots[i] << '\n';
    	}
    	fout.close();
	}

	std::cout<<"done all"<<std::endl;
}



int main(int argc, char** argv){
	std::string query_file, output_file, decision_graph_file;
	float density_cutoff, dist_cutoff;
	unsigned K, L, Lnn, num_threads, Lbuild, max_degree, Dbrute;
	po::options_description desc{"Arguments"};
 	try {
 		desc.add_options()("K",
                       po::value<unsigned>(&K)->required(),
                       "Number of neighbors for density computation");
 		desc.add_options()("L",
                       po::value<unsigned>(&L)->required(),
                       "L parameter for density computation");
 		desc.add_options()("Lnn",
                       po::value<unsigned>(&Lnn)->required(),
                       "L parameter for dependent point");
 		desc.add_options()("num_threads",
                       po::value<unsigned>(&num_threads)->default_value(4),
                       "number of threads");
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
	    desc.add_options()("Dbrute",
				       po::value<unsigned>(&Dbrute)->default_value(0),
				       "Number of points to bruteforce dependent point distance");
	    desc.add_options()("Lbuild",
                       po::value<unsigned>(&Lbuild)->default_value(100),
                       "L parameter for graph construction");
	    desc.add_options()("max_degree",
                       po::value<unsigned>(&max_degree)->default_value(64),
                       "max_degree of graph");
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
	dpc(K, L, Lnn, num_threads, density_cutoff, dist_cutoff, Dbrute, query_file, output_file, decision_graph_file, Lbuild, max_degree);
}