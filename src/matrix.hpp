#include <torch/torch.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <map>

using namespace std::chrono;
using std::string;
using std::ofstream;
using std::vector;
using std::map;
using std::unordered_map;

namespace matrix {

    torch::Tensor matrix;
    torch::Device device = torch::kCPU;
    map<vector<int>, float> mapResults;
    unordered_map<int, torch::Tensor> mapCache;
    float threshold;

    void setDevice(torch::Device );
    void initMatrix(int , int );
    void setThreshold(float );
    void setRandomSeed(int );
    void checkCondition(int , const vector<int>& );
    void writeResults(const string& );

} // matrix

void matrix::setDevice(torch::Device dev) {
    device = dev;
}

void matrix::initMatrix(int n, int m) {
    mapResults = map<vector<int>, float>();
    mapCache = unordered_map<int, torch::Tensor>();
    matrix = torch::rand({n, m}).to(device);
}

void matrix::setThreshold(float th) {
    threshold = th;
}

void matrix::setRandomSeed(int seed) {
    torch::manual_seed(seed);
}

void matrix::checkCondition(
        int conditionNum,
        const vector<int>& vecParentNodes) {
    torch::Tensor resVector;
    if (mapCache.find(conditionNum) != mapCache.end()) {
        resVector = mapCache[conditionNum];
    } else {
        resVector = matrix.slice(1, conditionNum, conditionNum + 1).to(device).reshape({1, -1})
                .mm(matrix.slice(1, conditionNum + 1).to(device)).to(device);
        mapCache[conditionNum] = resVector;
    }
    for (int i = 0; i < resVector.sizes()[1]; i++) {
        auto updVecParentNodes = vector<int>(vecParentNodes);
        updVecParentNodes.emplace_back(conditionNum + i + 1);
        if (resVector[0][i].item<float>() < threshold) {
            mapResults[updVecParentNodes] = resVector[0][i].item<float>();
        } else {
            checkCondition(
                    conditionNum + i + 1,
                    updVecParentNodes);
        }
    }
}

void matrix::writeResults(const string& fileName) {
    auto vec2str = [&](const vector<int>& vec) -> string {
        string s = "[";
        for (auto& item : vec) {
            s += std::to_string(item);
            s += ", ";
        }
        s[s.size() - 2] = ']';
        s[s.size() - 1] = ';';
        return s;
    };
    if (fileName == string("stdout")) {
        for (auto& [key, value] : matrix::mapResults) {
            std::cout << vec2str(key) << value << std::endl;
        }
    } else {
        ofstream fResults(fileName);
        for (auto& [key, value] : matrix::mapResults) {
            fResults << vec2str(key) << value << std::endl;
        }
        fResults.close();
    }
}