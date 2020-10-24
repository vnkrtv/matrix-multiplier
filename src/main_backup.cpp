// #include <qt5/QtConcurrent/QtConcurrent>
#include "configurator.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <map>

using namespace std::chrono;
using std::string;
using std::vector;
using std::map;

torch::Tensor matrix;
map<vector<int>, float> mapResults;

int conditionsCount;
int timestampsCount;
float threshold;

void checkConditions(
        int conditionNum,
        const vector<int>& vecParentNodes) {
    auto torchVec = matrix.slice(1, conditionNum, conditionNum + 1).reshape({1, -1});
    auto subMatrix = matrix.slice(1, conditionNum + 1);

    auto resVector = torchVec.mm(subMatrix);
    std::cout << conditionNum << ": " << vecParentNodes.size() << std::endl;
    for (int i = 0; i < resVector.sizes()[1]; i++) {
        auto updVecParentNodes = vector<int>(vecParentNodes);
        updVecParentNodes.emplace_back(conditionNum + i + 1);
        if (resVector[0][i].item<float>() < threshold) {
            mapResults[updVecParentNodes] = resVector[0][i].item<float>();
        } else {
            checkConditions(conditionNum + i + 1, updVecParentNodes);
        }
    }
}

int main(int argc, char const *argv[]) {
    // args::parse<Args>(argc, argv);
    conditionsCount = 5;
    timestampsCount = 5;
    threshold = 0;

//    map<vector<int>, float> mapResults;
    matrix = genTensor(timestampsCount, conditionsCount); //.cuda();
    for (int i = 0; i < conditionsCount; i++) {
        auto vecParents = vector<int>({i});
        checkConditions(i, vecParents);
    }
    for (auto& [key, value] : mapResults) {
        std::cout << key.size() << value << std::endl;
    }

    return 0;
}
