#include "matrix.hpp"

#include <qt5/QtConcurrent/QtConcurrent>

/*
using std::pair;

map<pair<int, int>, int> MapFunction(const pair<torch::Tensor&, pair<int, int>> & InterData) {



    map<pair<int, int>, int> mapEdges;
    auto copiedGraph = Graph(InterData.first.getEdgesVec());
    auto mapIt = InterData.first._mapNodes.begin();
    for (auto& it : InterData.second) {
        ShortestParts(copiedGraph, mapIt->second, mapEdges);
    }
    return mapEdges;
}

void ReduceFunction(map<pair<int, int>, int>& Results, const map<pair<int, int>, int>& InterData) {
    for (auto& it : InterData)
        Results[it.first] = it.second;
}*/
