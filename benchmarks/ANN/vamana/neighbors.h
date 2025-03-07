// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "common/geometry.h"
#include "../utils/NSGDist.h"  
#include "../utils/types.h"
#include "index.h"
#include "../utils/beamSearch.h"

extern bool report_stats;

template<typename T>
void ANN(parlay::sequence<Tvec_point<T>*> &v, int k, int maxDeg, int beamSize, double alpha, 
  parlay::sequence<Tvec_point<T>*> &q) {
  parlay::internal::timer t("ANN",report_stats); 
  {
    unsigned d = (v[0]->coordinates).size();
    using findex = knn_index<T>;
    findex I(maxDeg, beamSize, k, alpha, d);
    I.build_index(v, true);
    t.next("Built index");
    I.searchNeighbors(q, v);
    t.next("Found nearest neighbors");
    if(report_stats){
      //average numbers of nodes searched using beam search
      auto s = parlay::delayed_seq<size_t>(v.size(), [&] (size_t i) {return v[i]->cnt;});
      size_t i = parlay::max_element(s) - s.begin();
      size_t sum = parlay::reduce(s);
      std::cout << "max nodes searched = " << s[i] 
    << ", average nodes searched = " << sum/((double) v.size()) << std::endl;
      //average out-degree of graph
    auto od = parlay::delayed_seq<size_t>(v.size(), [&] (size_t i) {return v[i]->out_nbh.size();});
      size_t sum1 = parlay::reduce(od);
      std::cout << "Average graph out-degree = " << sum1/((double) v.size()) << std::endl;
      t.next("stats");
    }
    // for(int i=0; i<4*d; i++) std::cout << (int) v[0]->coordinates[i] << std::endl; 
    // std::cout << "next vector" << std::endl; 
    // for(int i=0; i<4*d; i++) std::cout << (int) v[v.size()]->coordinates[i] << std::endl; 
  };
}


template<typename T>
void ANN(parlay::sequence<Tvec_point<T>*> v, int k, int maxDeg, int beamSize, double alpha) {
  parlay::internal::timer t("ANN",report_stats); 
  { 
  
    unsigned d = (v[0]->coordinates).size();
    using findex = knn_index<T>;
    findex I(maxDeg, beamSize, k, alpha, d);
    I.build_index(v);
    t.next("Built index");
    if(report_stats){
      //average numbers of nodes searched using beam search
      auto s = parlay::delayed_seq<size_t>(v.size(), [&] (size_t i) {return v[i]->cnt;});
      size_t i = parlay::max_element(s) - s.begin();
      size_t sum = parlay::reduce(s);
      std::cout << "max nodes searched = " << s[i] 
    << ", average nodes searched = " << sum/((double) v.size()) << std::endl;
      //average out-degree of graph
    auto od = parlay::delayed_seq<size_t>(v.size(), [&] (size_t i) {return v[i]->out_nbh.size();});
      size_t sum1 = parlay::reduce(od);
      std::cout << "Average graph out-degree = " << sum1/((double) v.size()) << std::endl;
      t.next("stats");
    }
    // std::cout << "first vector: " << std::endl;
    // std::cout << "[";
    // for(int i=0; i<4*d; i++){
    //   std::cout << (int) v[0]->coordinates[i] << ", ";
    // } 
    // std::cout << "]" << std::endl; 
    // std::cout << "second vector: " << std::endl;
    // std::cout << "[";
    // for(int i=0; i<4*d; i++){
    //   std::cout << (int) v[1]->coordinates[i] << ", ";
    // } 
    // std::cout << "]" << std::endl; 
    // std::cout << "next vector" << std::endl; 
    // for(int i=0; i<4*d; i++) std::cout << (int) v[v.size()-1]->coordinates[i] << std::endl; 
  };
}
