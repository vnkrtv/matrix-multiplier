#include "matrix.hpp"

#include <qt5/QtConcurrent/QtConcurrent>

namespace mapReduce {

    struct MatrixProcessor {
        torch::Tensor _matrix;
        map<vector<int>, float> _mapResults;
        unordered_map<int, torch::Tensor> _mapCache;

        MatrixProcessor() {
            _matrix = matrix::matrix;
            _mapResults = map<vector<int>, float>();
            _mapCache = unordered_map<int, torch::Tensor>();
        }

    };

    void checkConditionMapReduceRecursive(int ,  const vector<int>& , MatrixProcessor& );
    void checkConditionMapReduce(int ,  const vector<int>& , MatrixProcessor& , queue<pair<int, vector<int>>>& );
    void Reduce(map<vector<int>, float>& , const map<vector<int>, float>& );
    map<vector<int>, float> MapRecursive(const vector<int>& );
    map<vector<int>, float> Map(const vector<int>& );
    map<vector<int>, float> MapReduce(int , bool );

} // mapReduce

void mapReduce::checkConditionMapReduceRecursive(
        int conditionNum,
        const vector<int>& vecParentNodes,
        MatrixProcessor& processor) {
    torch::Tensor resVector;
    if (processor._mapCache.find(conditionNum) != processor._mapCache.end()) {
        resVector = processor._mapCache[conditionNum];
    } else {
        resVector = processor._matrix.slice(1, conditionNum, conditionNum + 1).to(matrix::device).reshape({1, -1})
                .mm(processor._matrix.slice(1, conditionNum + 1).to(matrix::device)).to(matrix::device);
        processor._mapCache[conditionNum] = resVector;
    }
    for (int i = 0; i < resVector.sizes()[1]; i++) {
        auto updVecParentNodes = vector<int>(vecParentNodes);
        updVecParentNodes.emplace_back(conditionNum + i + 1);
        if (resVector[0][i].item<float>() < matrix::threshold) {
            processor._mapResults[updVecParentNodes] = resVector[0][i].item<float>();
        } else {
            checkConditionMapReduceRecursive(
                    conditionNum + i + 1,
                    updVecParentNodes,
                    processor);
        }
    }
}

void mapReduce::checkConditionMapReduce(
        int conditionNum,
        const vector<int>& vecParentNodes,
        MatrixProcessor& processor,
        queue<pair<int, vector<int>>>& qTasks) {
    torch::Tensor resVector;
    if (processor._mapCache.find(conditionNum) != processor._mapCache.end()) {
        resVector = processor._mapCache[conditionNum];
    } else {
        resVector = processor._matrix.slice(1, conditionNum, conditionNum + 1).to(matrix::device).reshape({1, -1})
                .mm(processor._matrix.slice(1, conditionNum + 1).to(matrix::device)).to(matrix::device);
        processor._mapCache[conditionNum] = resVector;
    }
    for (int i = 0; i < resVector.sizes()[1]; i++) {
        auto updVecParentNodes = vector<int>(vecParentNodes);
        updVecParentNodes.emplace_back(conditionNum + i + 1);
        if (resVector[0][i].item<float>() < matrix::threshold) {
            processor._mapResults[updVecParentNodes] = resVector[0][i].item<float>();
        } else {
            qTasks.emplace(conditionNum + i + 1, updVecParentNodes);
        }
    }
}

map<vector<int>, float> mapReduce::MapRecursive(const vector<int> & vecInnerData) {
    auto processor = MatrixProcessor();
    for (auto& item : vecInnerData) {
        auto vecParents = vector<int>({item});
        checkConditionMapReduceRecursive(item, vecParents, processor);
    }
    return processor._mapResults;
}

map<vector<int>, float> mapReduce::Map(const vector<int> & vecInnerData) {
    auto qTasks = queue<pair<int, vector<int>>>();
    for (auto& conditionNum : vecInnerData) {
        auto vecParents = vector<int>({conditionNum});
        qTasks.emplace(std::make_pair(conditionNum, vecParents));

    }
    auto processor = MatrixProcessor();
    while (!qTasks.empty()) {
        auto pairTask = qTasks.front();
        checkConditionMapReduce(pairTask.first, pairTask.second, processor, qTasks);
        qTasks.pop();
    }
    return processor._mapResults;
}

void mapReduce::Reduce(map<vector<int>, float>& mapResults, const map<vector<int>, float>& mapInnerData) {
    for (auto& it : mapInnerData)
        mapResults[it.first] = it.second;
}

map<vector<int>, float> mapReduce::MapReduce(int conditionsCount, bool recursive) {
    auto idealThreadCount = QThread::idealThreadCount();
    vector<vector<int>> vecData;

    int first = 0;
    int delta = conditionsCount < idealThreadCount ? 1 : round(float(conditionsCount) / idealThreadCount);
    int last = delta;
    for (int i = 0; i < idealThreadCount; i++) {
        auto vecItem = vector<int>();
        for (int i = first; i < last; i++) {
            vecItem.emplace_back(i);
        }
        vecData.emplace_back(vecItem);
        first = last;
        last = last + delta < conditionsCount ? last + delta : conditionsCount;
    }

    if (recursive) {
        return QtConcurrent::blockingMappedReduced(vecData, MapRecursive, Reduce);
    } else {
        return QtConcurrent::blockingMappedReduced(vecData, Map, Reduce);
    }
}