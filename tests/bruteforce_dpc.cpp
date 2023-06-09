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
#include "pargeo/point.h"


namespace po = boost::program_options;


std::vector<std::vector<float>> read_file(std::string& query_file){
	std::ifstream fin(query_file);
	std::string line;
	std::vector<std::vector<float>> points;
	while(std::getline(fin, line)){
		std::stringstream line_stream(line);
		points.push_back(std::vector<float>());
		float x;
		while(line_stream >> x){
			points.back().push_back(x);
		}
	}
	return points;
}

struct UFDS{
	std::vector<size_t> finder;
	UFDS(){}
	UFDS(size_t n){
		finder = std::vector<size_t>(n);
		std::iota(finder.begin(), finder.end(), 0);
	}
	size_t find(size_t a){
		size_t olda = a;
		for(;finder[a]!=a;a=finder[a]);
		for(;finder[olda]!=a;){
			size_t newa = finder[olda];
			finder[olda] = a;
			olda = newa;
		} 
		return a;
	}
	void merge(size_t a,size_t b){
		size_t fa = find(a);
		size_t fb = find(b);
		if(fa == fb) return;
		finder[fa] = fb;
	}
};

void dpc(const unsigned K, float density_cutoff, float dist_cutoff, std::string& query_file, std::string& output_file, std::string& decision_graph_file){
	std::vector<std::vector<float>> points = read_file(query_file);
	size_t n = points.size();
	size_t d = points[0].size();

	auto calc_dist = [&](std::vector<float>& a, std::vector<float>& b){
		float sum = 0;
		for(size_t i=0;i<d;i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
		return sqrt(sum);
	};

	std::vector<float> densities(n);

	for(size_t i=0; i<n; i++) {
		std::vector<float> dists(n);
		for(size_t j=0; j<n; j++) dists[j] = calc_dist(points[i], points[j]);
		std::nth_element(dists.begin(), dists.begin()+K-1, dists.end());
		densities[i] = 1/dists[K-1];
	}

	std::vector<size_t> dep_ptrs(n);
	std::vector<float> dep_dists(n);

	for(size_t i=0; i<n; i++) {
		std::vector<float> dists(n, std::numeric_limits<float>::max());
		for(size_t j=0; j<n; j++) 
			if(densities[j] > densities[i])
				dists[j] = calc_dist(points[i], points[j]);
		float m_dist = std::numeric_limits<float>::max();
		size_t id = n;
		for(size_t j=0; j<n; j++){
			if(dists[j] < m_dist){
				m_dist = dists[j];
				id = j;
			}
		}
		dep_ptrs[i] = id;
		dep_dists[i] = m_dist;
	}

	if(decision_graph_file != ""){    	
    	std::ofstream fout(decision_graph_file);
    	for (size_t i = 0; i < n; i++){
    		fout << densities[i] << " " << dep_dists[i] << '\n';
    	}
    }

	
	UFDS finder(n);
	for(size_t i=0; i<n; i++){
		if(densities[i]<=density_cutoff || dep_dists[i]<=dist_cutoff){
			finder.merge(i, dep_ptrs[i]);
		}
	}

	std::vector<int> cluster(n);
	for(size_t i=0; i<n; i++) cluster[i] = finder.find(i);

    if(output_file != ""){
    	std::ofstream fout(output_file);
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
	dpc(2, density_cutoff, dist_cutoff, query_file, output_file, decision_graph_file);
}