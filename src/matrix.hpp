#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <map>

using namespace std::chrono;
using std::string;
using std::vector;
using std::map;
using std::unordered_map;

torch::Tensor genTensor(int n, int m, int seed = 10) {
    torch::manual_seed(seed);
    return torch::rand({n, m});
}

void checkConditions(
        torch::Tensor& matrix,
        int conditionNum,
        const vector<int>& vecParentNodes,
        map<vector<int>, float>& mapResults,
        unordered_map<int, torch::Tensor>& mapCache,
        int conditionsCount,
        float threshold) {
    torch::Tensor resVector;
    if (mapCache.find(conditionNum) != mapCache.end()) {
        resVector = mapCache[conditionNum];
    } else {
        auto torchVec = matrix.slice(1, conditionNum, conditionNum + 1).reshape({1, -1});
        auto subMatrix = matrix.slice(1, conditionNum + 1);
        resVector = torchVec.mm(subMatrix);
        mapCache[conditionNum] = resVector;
    }
    for (int i = 0; i < resVector.sizes()[1]; i++) {
        auto updVecParentNodes = vector<int>(vecParentNodes);
        updVecParentNodes.emplace_back(conditionNum + i + 1);
        if (resVector[0][i].item<float>() < threshold) {
            mapResults[updVecParentNodes] = resVector[0][i].item<float>();
        } else {
            checkConditions(
                    matrix,
                    conditionNum + i + 1,
                    updVecParentNodes,
                    mapResults,
                    mapCache,
                    conditionsCount,
                    threshold);
        }
    }
}
