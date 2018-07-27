#include "mex.h"
#include <iostream>
#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/kernighan-lin.hxx"
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
using namespace std;

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    
    
  double* nNodesP = mxGetPr(prhs[0]);
  double* nEdgesP = mxGetPr(prhs[1]);
   
  int nNodes = nNodesP[0];
  int nEdges = nEdgesP[0];
  std::cout << nNodes << std::endl;
  std::cout << nEdges<< std::endl;
   
  double* nLinks= mxGetPr(prhs[2]);
   
  andres::graph::Graph<> graph;
  graph.insertVertices(nNodes);
  std::vector<double> weights(nEdges);

  for (int iEdge=0;iEdge<=nEdges-1;iEdge++){
    graph.insertEdge(nLinks[iEdge],nLinks[iEdge+nEdges]);
    weights[iEdge] = nLinks[iEdge+2*nEdges];
  }
    
  std::vector<char> edge_labels(graph.numberOfEdges());
   andres::graph::multicut::kernighanLin(graph, weights, edge_labels,edge_labels);
  //andres::graph::multicut::ilp<andres::ilp::Gurobi>(graph, weights, edge_labels, edge_labels);

  std::vector<pair<int,int> > entryList;


  
  for (int i=0; i<nNodes;i++){
    for(int j=i+1; j<nNodes;j++){
      std::pair<bool,std::size_t> graphRes =  graph.findEdge(i,j);
      if (graphRes.first){
	bool label=  edge_labels[graphRes.second];
	if (label == 0){
	  entryList.push_back(std::make_pair(i,j));
	  //	  resList[i][j]=1;
	} 
      }
    }
  }
  plhs[0] = mxCreateDoubleMatrix(1,entryList.size()*2,mxREAL);

  int linIdx = 0;
  double* resP = mxGetPr(plhs[0]);
  for(const pair<int,int> &entry : entryList){

    resP[linIdx] = entry.first;
        resP[linIdx+1] = entry.second;
	linIdx = linIdx+2;

  }
//  
//  for (int i = 0; i< entryList.size()*2; i++){
//    for (int j = 0; j< nNodes; j++){
//      entryList.
//      resP[nNodes*i+j]= resList[i][j;
//    }
//  }
}
