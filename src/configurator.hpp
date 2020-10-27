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
    bool _mapReduce = false;
    bool _recursively = false;
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
        f(_seed, "--seed", "-s", args::help("Random seed (21 by default)"));
        f(_resultsFileName, "--output", "-o", args::help("Result file name (stdout by default)"));
        f(_trainOnCuda, "--cuda", args::help("Training on GPU with CUDA"), args::set(true));
        f(_mapReduce, "--map-reduce", args::help("Use map reduce"), args::set(true));
        f(_recursively, "--recursion", args::help("Algorithm with recursion"), args::set(true));
    }

    void run() {
        std::cout << "Conditions count: " << _conditionsCount << std::endl;
        std::cout << "Timestamps count: " << _timestampsCount << std::endl;
        std::cout << "Matrix " << _timestampsCount << "x" << _conditionsCount << std::endl;
        std::cout << "Threshold: " << _threshold << std::endl;
        std::cout << "Random seed: " << _seed << std::endl;
        std::cout << "Results file name: " << _resultsFileName << std::endl << std::endl;

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

        if (_mapReduce) {
            std::cout << "Using MapReduce computing: ideal thread count - " << QThread::idealThreadCount() << std::endl;
        }
        if (_recursively) {
            std::cout << "Using recursive realization" << std::endl;
        }

        matrix::initMatrix(_timestampsCount, _conditionsCount);
        matrix::setThreshold(_threshold);
        matrix::setRandomSeed(_seed);

        auto start = steady_clock::now();
        
        if (!_recursively && !_mapReduce) {
            auto qTasks = queue<pair<int, vector<int>>>();
            for (int conditionNum = 0; conditionNum < _conditionsCount; conditionNum++) {
                auto vecParents = vector<int>({conditionNum});
                qTasks.emplace(std::make_pair(conditionNum, vecParents));

            }
            while (!qTasks.empty()) {
                auto pairTask = qTasks.front();
                matrix::checkCondition(pairTask.first, pairTask.second, qTasks);
                qTasks.pop();
            }
        }
        if (_recursively && !_mapReduce) {
            for (int conditionNum = 0; conditionNum < _conditionsCount; conditionNum++) {
                auto vecParents = vector<int>({conditionNum});
                matrix::checkConditionRecursive(conditionNum, vecParents);
            }
        }

        if (!_recursively && _mapReduce) {
            matrix::mapResults = mapReduce::MapReduce(_conditionsCount, false);
        }
        if (_recursively && _mapReduce) {
            matrix::mapResults = mapReduce::MapReduce(_conditionsCount, true);
        }

        double duration = duration_cast<milliseconds>(steady_clock::now() - start).count();

        matrix::writeResults(_resultsFileName);
        std::cout << "Finished in " << duration << " milliseconds. Write results to " << _resultsFileName << std::endl;
    }
};
