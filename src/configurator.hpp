#include <iostream>
#include <string>

#include "args.hpp"
#include "mapreduce.hpp"

using namespace std::chrono;
using std::string;

struct Args {
    int _conditionsCount;
    int _timestampsCount;
    float _threshold;
    int _seed = 21;
    bool _trainOnCuda = false;
    string _resultsFileName = "stdout";


    Args() = default;

    static const char* help() {
        return "Calculating matrices";
    }

    template<class F>
    void parse(F f) {
        f(_conditionsCount, "--conditions-count", "-n", args::help("Conditions count"), args::required());
        f(_timestampsCount, "--timestamps-count", "-m", args::help("Timestamps count"), args::required());
        f(_threshold, "--threshold", "-t", args::help("Threshold"), args::required());
        f(_seed, "--seed", "-s", args::help("Random seed"));
        f(_resultsFileName, "--output", "-o", args::help("Result file name (stdout by default)"));
        f(_trainOnCuda, "--cuda", args::help("Training on GPU with CUDA"), args::set(true));
    }

    void run() {
        std::cout << "Conditions count: " << _conditionsCount << std::endl;
        std::cout << "Timestamps count: " << _timestampsCount << std::endl;
        std::cout << "Matrix " << _timestampsCount << "x" << _conditionsCount << std::endl;
        std::cout << "Threshold: " << _threshold << std::endl;
        std::cout << "Random seed: " << _seed << std::endl;
        std::cout << "Results file name: " << _resultsFileName << std::endl;

        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available() && _trainOnCuda) {
            std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            matrix::setDevice(torch::kCUDA);
        } else {
            if (!torch::cuda::is_available()) {
                std::cout << "CUDA is not available." << std::endl;
            }
            matrix::setDevice(torch::kCPU);
            std::cout << "Training on CPU." << std::endl;
        }

        matrix::initMatrix(_timestampsCount, _conditionsCount);
        matrix::setThreshold(_threshold);
        matrix::setRandomSeed(_seed);

        auto start = steady_clock::now();
        for (int i = 0; i < _conditionsCount; i++) {
            auto vecParents = vector<int>({i});
            matrix::checkCondition(i, vecParents);
        }
        double duration = duration_cast<milliseconds>(steady_clock::now() - start).count();

        matrix::writeResults(_resultsFileName);
        std::cout << "Finished in " << duration << " milliseconds. Write results to " << _resultsFileName << std::endl;
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
