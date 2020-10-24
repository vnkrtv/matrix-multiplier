#include <iostream>
#include <string>

#include "args.hpp"
#include "mapreduce.hpp"

using std::string;

struct Args {
    int _conditionsCount;
    int _timestampsCount;
    float _threshold;
    bool _trainOnCuda = false;


    Args() = default;

    static const char* help() {
        return "Calculating matrices";
    }

    template<class F>
    void parse(F f) {
        f(_conditionsCount, "--conditions-count", "-n", args::help("Conditions count"), args::required());
        f(_timestampsCount, "--timestamps-count", "-m", args::help("Timestamps count"), args::required());
        f(_threshold, "--threshold", "-t", args::help("Threshold"), args::required());
        f(_trainOnCuda, "--cuda", args::help("Training in cuda"));
    }

    void run() {
        std::cout << "Conditions count: " << _conditionsCount << std::endl;
        std::cout << "Timestamps count: " << _timestampsCount << std::endl;
        std::cout << "Matrix " << _timestampsCount << "x" << _timestampsCount << std::endl;
        std::cout << "Threshold: " << _threshold << std::endl;

        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available() && _trainOnCuda) {
            std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::kCUDA;
        } else {
            std::cout << "Training on CPU." << std::endl;
        }

        map<vector<int>, float> mapResults;
        unordered_map<int,torch::Tensor> mapCache;
        auto matrix = genTensor(_timestampsCount, _conditionsCount).to(torch::kCUDA);
        for (int i = 0; i < _conditionsCount; i++) {
            auto vecParents = vector<int>({i});
            checkConditions(
                    matrix,
                    i,
                    vecParents,
                    mapResults,
                    mapCache,
                    _conditionsCount,
                    _threshold);
        }
        auto vec2str = [&](const vector<int>& vec) -> string {
            string s = "[";
            for (auto& item : vec) {
                s += std::to_string(item);
                s += ", ";
            }
            s[s.size() - 2] = ']';
            return s;
        };
        for (auto& [key, value] : mapResults) {
            std::cout << vec2str(key) << "=> " << value << std::endl;
        }
        /*
        auto matr = genTensor()


        auto availableThreadsCount = QThread::idealThreadCount();
        vector<pair<Graph, map<int, Node*>>> VData;

        int first = 0;
        int delta = copiedGraph.getNodesCount() < availableThreadsCount ? 1 : round(float(copiedGraph.getNodesCount()) / availableThreadsCount); // copiedGraph.getNodesCount() / Ithread;
        int last = delta;
        for (int i = 0; i < availableThreadsCount; i++) {
            auto item = std::make_pair(copiedGraph, map<int, Node*>());
            auto mapIt = copiedGraph._mapNodes.begin();
            for (int i = 0; i < first; i++, mapIt++);
            for (int i = 0; i < last; i++, mapIt++) {
                if (mapIt == copiedGraph._mapNodes.end()) {
                    break;
                }
                item.second[mapIt->first] = mapIt->second;
            }

            VData.emplace_back(item);
            first = last;
            last = last + delta < copiedGraph.getNodesCount() ? last + delta : copiedGraph.getNodesCount();
        }

        auto mapEdges = QtConcurrent::blockingMappedReduced(VData, MapFunction, ReduceFunction);
        */
    }
};
