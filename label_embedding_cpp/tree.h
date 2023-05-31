#include <algorithm>
#include <armadillo>
#include <chrono>
#include <random>
#include <stack>
#include "dtutil.h"
#include <vector>

#if defined(DEBUG)
#define DEBUG_MSG(str) do {std::cerr << "DEBUG: " << str << std::endl;} \
    while (false)
#else
#define DEBUG_MSG(str) do { } while (false)
#endif

// define the min increment of featur value
#define MIN_FEATURE_INC 1e-6

template <typename DATATYPE, typename EMBEDTYPE>
class DTRegressor {
    public:
    DTRegressor(
        int minSamplesSplit_,
        int minSamplesLeaf_,
        int maxFeatures_,
        int maxDepth_,
        double stopCriterion_,
        unsigned seed
    ):
        minSamplesSplit(minSamplesSplit_),
        minSamplesLeaf(minSamplesLeaf_),
        maxFeatures(maxFeatures_),
        maxDepth(maxDepth_),
        stopCriterion(stopCriterion_),
        generator(seed) 
    {}

    DTRegressor(
        int minSamplesSplit_,
        int minSamplesLeaf_,
        int maxFeatures_,
        int maxDepth_,
        double stopCriterion_
    ):
        DTRegressor(
            minSamplesSplit_,
            minSamplesLeaf_,
            maxFeatures_,
            maxDepth_,
            stopCriterion_,
            (unsigned)
                std::chrono::system_clock::now().time_since_epoch().count()
        )
    {}

    void train(
        const DATATYPE& X,
        const arma::uvec& y,
        const EMBEDTYPE& embedM
    );

    void train(
        const DATATYPE& X,
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        const unsigned seed
    );
    
    void train(
        const DATATYPE& X,
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights
    );

    void predict(const DATATYPE& X, EMBEDTYPE& preds);

    void printTree();
 
     private:
    //hyperparameters of dt:
    int minSamplesSplit;  
    int minSamplesLeaf;
    int maxFeatures;
    int maxDepth;
    double stopCriterion; // stop if MSE is smaller than this on a node

    std::default_random_engine generator;

    // array of nodes, nodes[0] is the root of the tree
    std::vector<DTNode> nodesVec;
    
    // store predictions
    std::vector<std::vector<double>> predsVec;

    void calcEmbedSum(
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        const arma::uvec& indices,
        const size_t begin,
        const size_t end,  // end is always exclusive
        EMBEDTYPE& targetSum,
        double& weightSum,
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache
    );

    double __calcMSE(
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        const arma::uvec& indices,
        const size_t begin,
        const size_t end,  // end is always exclusive
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache
    );

    /*
    Calculate error and store left and right sum for future update. This 
    function should only be called once per feature. When looking for the 
    optimal threshold, the function ___updateGain should be called.
    */
    double ___calcGain(
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        const arma::uvec& indices,
        const size_t begin,
        const size_t end,  // end is always exclusive
        const size_t pos,  // i in left iff i < pos
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache 
    );

    /*
    Update the gain when split position changes from oldPos to newPos.
    */
    double ___updateGain(
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        const arma::uvec& indices,
        const size_t begin,
        const size_t end,  // end is always exclusive
        const size_t oldPos,  //position of previous split
        const size_t newPos,  //position of new split
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache 
    );

    /*
    This function should be called by a parent node when one of its children
    is to be made leaf.
    */ 
    void __makeLeaf(
        const size_t nodeIdx,
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::uvec& indices,
        const size_t begin,
        const size_t end,
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache 
    );

    template <typename Ele>
    void __searchThreshold(  // will be called it _buildNode 
        const arma::Mat<Ele>& X,
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        arma::uvec& indices,
        const size_t begin,
        const size_t end,
        std::vector<size_t>& features,
        const size_t currFeatIdx,
        size_t& bestFeatSplit,
        double& bestFeatGain,
        size_t& nConstants,
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache
    );

    template <typename Ele>
    void __searchThreshold(  // will be called it _buildNode 
        const arma::SpMat<Ele>& X,
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        arma::uvec& indices,
        const size_t begin,
        const size_t end,
        std::vector<size_t>& features,
        const size_t currFeatIdx,
        size_t& bestFeatSplit,
        double& bestFeatGain,
        double& bestFeatThreshold,
        size_t& nConstants,
        int& isSorted,
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache
    );

    template <typename Ele>
    void _buildNode(
        const nodeBuilder& builder,
        const arma::Mat<Ele>& X,
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        arma::uvec& indices,
        std::stack<nodeBuilder>& buildersStack,
        std::vector<size_t>& features,
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache 
    );

    template <typename Ele>
    void _buildNode(
        const nodeBuilder& builder,
        const arma::SpMat<Ele>& X,
        const arma::uvec& y,
        const EMBEDTYPE& embedM,
        const arma::vec& weights,
        arma::uvec& indices,
        std::stack<nodeBuilder>& buildersStack,
        std::vector<size_t>& features,
        treeTrainCache<arma::fmat, EMBEDTYPE>& cache 
    );
};

//Implementation:
template <typename DT, typename ET>
inline void DTRegressor<DT, ET>::calcEmbedSum(
    const arma::uvec& y,
    const ET& embedM,
    const arma::vec& weights,
    const arma::uvec& indices,
    const size_t begin,
    const size_t end,  // end is always exclusive
    ET& targetSum,
    double& weightSum,
    treeTrainCache<arma::fmat, ET>& cache
) {
    size_t nData = end - begin;
    size_t nClasses = embedM.n_cols;
    size_t nEmbed = embedM.n_rows;
    // N: num data, C: num classes, E: embedding dim
    // count-and-matmul takes N + 2CE - E
    // brute force takes 2NE + N
    if (nData < nClasses) {
        targetSum.zeros();
        weightSum = 0;
        for (size_t i = begin; i < end; i++){ 
            targetSum += weights(y(indices(i))) *  embedM.col(y(indices(i)));
            weightSum += weights(y(indices(i)));
        }
        return; 
    }
    cache.labelCounter.zeros();
    for(size_t i = begin; i < end; i++) 
        cache.labelCounter(y(indices(i))) += weights(y(indices(i)));
    targetSum = embedM * cache.labelCounter;
    weightSum = arma::accu(cache.labelCounter);
}

template <typename DT, typename ET>
inline double DTRegressor<DT, ET>::__calcMSE(
    const arma::uvec& y,
    const ET& embedM,
    const arma::vec& weights,
    const arma::uvec& indices,
    const size_t begin,
    const size_t end,  // end is always exclusive
    treeTrainCache<arma::fmat, ET>& cache
) {
    size_t nData = end - begin;
    size_t nClasses = embedM.n_cols;
    size_t nEmbed = embedM.n_rows;
    // N: num data, C: num classes, E: embedding dim
    // count-and-matmul takes N + 2EC - E operations with precomputation
    // brute force takes 2NE operations with precomputation
    double sqSum = 0;
    if (nData < nClasses) {
        for (size_t i = begin; i < end; i++) 
            sqSum += weights(indices(i)) 
                * arma::accu(cache.embedMSq.col(y(indices(i))));
        return sqSum / cache.tW 
            - arma::accu(arma::square(cache.tSum)) / (cache.tW * cache.tW);
    }
    cache.labelCounter.zeros();
    for(size_t i = begin; i < end; i++)
        cache.labelCounter(y(indices(i))) += weights(y(indices(i)));
    sqSum = arma::accu(cache.embedMSq * cache.labelCounter);
    return sqSum / cache.tW 
        - arma::accu(arma::square(cache.tSum)) /(cache.tW * cache.tW);
}

template <typename DT, typename ET>
inline double DTRegressor<DT, ET>::___calcGain(
    const arma::uvec& y,
    const ET& embedM,
    const arma::vec& weights,
    const arma::uvec& indices,
    const size_t begin,
    const size_t end,  // end is always exclusive
    const size_t pos,  // i in left iff i < pos
    treeTrainCache<arma::fmat, ET>& cache
) {
    if ((pos - begin) < (end - pos)) {
        this->calcEmbedSum(
            y,
            embedM,
            weights,
            indices,
            begin,
            pos,
            cache.lSum,
            cache.lW,
            cache);
        cache.rSum = cache.tSum - cache.lSum;
        cache.rW = cache.tW - cache.lW; 
    }
    else {
        this->calcEmbedSum(
            y,
            embedM,
            weights,
            indices,
            pos,
            end,
            cache.rSum,
            cache.rW,
            cache);
        cache.lSum = cache.tSum - cache.rSum;
        cache.lW = cache.tW - cache.rW;
    }
    return arma::accu(arma::square(cache.lSum)) / cache.lW + 
        arma::accu(arma::square(cache.rSum)) / cache.rW;
}

template <typename DT, typename ET>
inline double DTRegressor<DT, ET>::___updateGain(
    const arma::uvec& y,
    const ET& embedM,
    const arma::vec& weights,
    const arma::uvec& indices,
    const size_t begin,
    const size_t end,  // end is always exclusive
    const size_t oldPos,  //position of previous split
    const size_t newPos,  //position of new split
    treeTrainCache<arma::fmat, ET>& cache 
) {
    // first update the sums
    if ((newPos - oldPos) <= (end - newPos)) {  // update left
        this->calcEmbedSum(
            y,
            embedM,
            weights,
            indices,
            oldPos,
            newPos,
            cache.lSum,
            cache.lW,
            cache
        );
        cache.rSum -= cache.lSum;
        cache.rW -= cache.lW;
        cache.lSum = cache.tSum - cache.rSum;
        cache.lW = cache.tW - cache.rW;
    }
    else {  // calculate right
        this->calcEmbedSum(
            y,
            embedM,
            weights,
            indices,
            newPos,
            end,
            cache.rSum,
            cache.rW,
            cache
        );
        cache.lSum = cache.tSum - cache.rSum;
        cache.lW = cache.tW - cache.rW;
    }
    int leftSize = newPos - begin;
    int rightSize = end - newPos;
    return arma::accu(arma::square(cache.lSum)) / cache.lW + 
        arma::accu(arma::square(cache.rSum)) / cache.rW;  
}

template <typename DT, typename ET>
inline void DTRegressor<DT, ET>::__makeLeaf(
    const size_t nodeIdx,
    const arma::uvec& y,
    const ET& embedM,
    const arma::uvec& indices,
    const size_t begin,
    const size_t end,
    treeTrainCache<arma::fmat, ET>& cache
) {
    this->nodesVec[nodeIdx].featureIdx = -1;  // this indicates a leaf node
    DEBUG_MSG("making leaf node of size " << (end - begin));
    this->nodesVec[nodeIdx].predIdx = this->predsVec.size();
    auto pred = arma::conv_to<std::vector<double>>::from(cache.tSum/cache.tW);
    this->predsVec.push_back(std::move(pred));
    DEBUG_MSG(
        "leaf node has been made with prediction value stored at "
        << this->nodesVec[nodeIdx].predIdx
    );
}

template <typename DT, typename ET>
template <typename Ele>
inline void DTRegressor<DT, ET>::__searchThreshold(
        const arma::Mat<Ele>& X,
        const arma::uvec& y,
        const ET& embedM,
        const arma::vec& weights,
        arma::uvec& indices,
        const size_t begin,
        const size_t end,
        std::vector<size_t>& features,
        const size_t currFeatIdx,
        size_t& bestFeatSplit,
        double& bestFeatGain,
        size_t& nConstants,
        treeTrainCache<arma::fmat, ET>& cache 
) {
    size_t currFeat = features[currFeatIdx];
    // copy the features and sort in a more cache-friendly manner
    for (int i = begin; i < end; i++) cache.Xf(i) = X(indices(i), currFeat);
    sort(cache.Xf.memptr() + begin, indices.memptr() + begin, end - begin);
    if (  // removing -1 in the next line is wrong but has better accuracy
        cache.Xf(begin + this->minSamplesLeaf -1) + MIN_FEATURE_INC
        > cache.Xf(end - this->minSamplesLeaf)
    ) {
        std::swap(features[nConstants], features[currFeatIdx]); 
        nConstants++;
        return;  // skip constant feature
    }
    bestFeatGain = -1; 
    // attempt to find a good initial position
    size_t initialPos = begin + this->minSamplesLeaf;
    while (
        (cache.Xf(initialPos) < (cache.Xf(initialPos-1) + MIN_FEATURE_INC))
        && (initialPos <= (end - this->minSamplesLeaf))
    )
        initialPos++;
    // check if a good initial position can be found
    if (initialPos > (end - this->minSamplesLeaf)) {
        std::swap(features[nConstants], features[currFeatIdx]); 
        nConstants++;
        return;
    }
    // calculate the gain at this point
    size_t dimE = embedM.n_rows;  // embedding dimension
    bestFeatSplit = initialPos;
    bestFeatGain = this->___calcGain(
        y,
        embedM,
        weights,
        indices,
        begin,
        end,
        bestFeatSplit,
        cache 
    );
    size_t lastPos = initialPos;
    // look for the best split
    for (size_t splitPos = initialPos+1;
        splitPos <= (end - this->minSamplesLeaf);
        splitPos++) {
        if (cache.Xf(splitPos) < (cache.Xf(splitPos-1) + MIN_FEATURE_INC)) continue;
        double currGain = this->___updateGain(
            y,
            embedM,
            weights,
            indices,
            begin,
            end,
            lastPos,
            splitPos,
            cache
        );
        lastPos = splitPos;
        if (currGain > bestFeatGain) {
            bestFeatGain = currGain;
            bestFeatSplit = splitPos; 
        }
    }
}

template <typename DT, typename ET>
template <typename Ele>
inline void DTRegressor<DT, ET>::__searchThreshold(
        const arma::SpMat<Ele>& X,
        const arma::uvec& y,
        const ET& embedM,
        const arma::vec& weights,
        arma::uvec& indices,
        const size_t begin,
        const size_t end,
        std::vector<size_t>& features,
        const size_t currFeatIdx,
        size_t& bestFeatSplit,
        double& bestFeatGain,
        double& bestFeatThreshold,
        size_t& nConstants,
        int& isSorted,
        treeTrainCache<arma::fmat, ET>& cache
) {
    size_t currFeat = features[currFeatIdx];
    size_t negEnd, posBegin;

    DEBUG_MSG("sparse split search: extracting non zeros");
    // extract the nonzero elements and sort them
    extractNNZ(
        X,
        indices,
        begin,
        end,
        currFeat,
        negEnd,
        posBegin,
        isSorted,
        cache
    );

    DEBUG_MSG("sparse split search: non zeros extracted");
    DEBUG_MSG("sparse split search: current features: ");
    DEBUG_MSG(cache.Xf.submat(begin, 0, end-1, 0));
    DEBUG_MSG("sparse split search: current indices: ");
    DEBUG_MSG(indices.subvec(begin, end-1));
    
    sort(cache.Xf.memptr() + begin, indices.memptr() + begin, negEnd - begin);
    if (posBegin < end) 
        sort(
            cache.Xf.memptr() + posBegin,
            indices.memptr() + posBegin,
            end - posBegin
        );
    DEBUG_MSG("sparse split search: non zeros sorted");
    DEBUG_MSG("sparse split search: current features: ");
    DEBUG_MSG(cache.Xf.submat(begin, 0, end-1, 0));
    DEBUG_MSG("sparse split search: current indices: ");
    DEBUG_MSG(indices.subvec(begin, end-1));
    // update the inverse map
    for (size_t i = begin; i < negEnd; i++) cache.row2Idx(indices(i)) = i;
    for (size_t i = posBegin; i < end; i++) cache.row2Idx(indices(i)) = i;
    DEBUG_MSG("sparse split search: inverse map updated");
    
    // removing -1 on the next line is incorrect but has better accuracy
    size_t leftMost = begin + this->minSamplesLeaf - 1;
    size_t rightMost = end - this->minSamplesLeaf;
    if (
        sparseGet(leftMost, negEnd, posBegin, cache) + MIN_FEATURE_INC
        > sparseGet(rightMost, negEnd, posBegin, cache)
    ) {
        std::swap(features[nConstants], features[currFeatIdx]); 
        nConstants++;
        return;  // skip constant feature
    }

    // add one or two zeros, same as sklearn
    if (negEnd < posBegin) {
        cache.Xf[--posBegin] = 0;
        if (negEnd != posBegin) cache.Xf[negEnd++] = 0;
    }

    bestFeatGain = -1; 
    // attempt to find a good initial position
    size_t initialPos = begin + this->minSamplesLeaf;
    if (sparseIsZero(initialPos, negEnd, posBegin)) initialPos = posBegin;

    size_t prevPos = initialPos - 1;
    if (initialPos == posBegin) prevPos = negEnd - 1;
    DEBUG_MSG("sparse split search: searching for initial positiong");
    while (
        (initialPos <= (end - this->minSamplesLeaf))
        && (cache.Xf(initialPos) < (cache.Xf(prevPos) + MIN_FEATURE_INC))
    ) sparseIncrement(initialPos, prevPos, negEnd, posBegin);
 
    // check if a good initial position can be found
    if (initialPos > (end - this->minSamplesLeaf)) {
        std::swap(features[nConstants], features[currFeatIdx]); 
        nConstants++;
        return;
    }

    DEBUG_MSG("sparse split search: initial position found");
    // calculate the gain at this point
    size_t dimE = embedM.n_rows;  // embedding dimension
    bestFeatSplit = initialPos;
    bestFeatGain = this->___calcGain(
        y,
        embedM,
        weights,
        indices,
        begin,
        end,
        bestFeatSplit,
        cache 
    );
    size_t lastPos = initialPos;
    bestFeatThreshold = 
        (sparseGet(initialPos-1, negEnd, posBegin, cache) 
        + sparseGet(initialPos, negEnd, posBegin, cache)) / 2.0;

    DEBUG_MSG("sparse split search: iterating through features");
    DEBUG_MSG("sparse split search: current features: ");
    DEBUG_MSG(cache.Xf.submat(begin, 0, end-1, 0));
    DEBUG_MSG("sparse split search: current indices: ");
    DEBUG_MSG(indices.subvec(begin, end-1));
    size_t splitPos = initialPos;
    while (splitPos < (end - this->minSamplesLeaf)) {
        sparseIncrement(splitPos, prevPos, negEnd, posBegin); 
        DEBUG_MSG(
            "sparse split search: evaluating between " << splitPos 
            << " and " << prevPos
        );
        if (cache.Xf(splitPos) < (cache.Xf(prevPos) + MIN_FEATURE_INC)) {
            //sparseIncrement(splitPos, prevPos, negEnd, posBegin); 
            continue;
        }
        double currGain = this->___updateGain(
            y,
            embedM,
            weights,
            indices,
            begin,
            end,
            lastPos,
            splitPos,
            cache
        );
        lastPos = splitPos;
        if (currGain > bestFeatGain) {
            bestFeatGain = currGain;
            bestFeatSplit = splitPos;
            float left = sparseGet(splitPos-1, negEnd, posBegin, cache);
            float right = sparseGet(splitPos, negEnd, posBegin, cache);
            bestFeatThreshold = (left + right) / 2.0;
        }
    }
    DEBUG_MSG("sparse split search: split search done");
}

template <typename DT, typename ET>
template <typename Ele>
inline void DTRegressor<DT, ET>::_buildNode(
    const nodeBuilder& builder,
    const arma::Mat<Ele>& X,
    const arma::uvec& y,
    const ET& embedM,
    const arma::vec& weights,
    arma::uvec& indices,
    std::stack<nodeBuilder>& buildersStack, 
    std::vector<size_t>& features,
    treeTrainCache<arma::fmat, ET>& cache
) {
    const int nodeIdx = builder.nodeIdx;
    const size_t begin = builder.begin;
    const size_t end = builder.end;
    const size_t currentDepth = builder.currentDepth;
    size_t nConstants = builder.nConstants;

    int bestSplitPos = -1;
    int bestFeat = -1;
    double bestThreshold = -1; 
    double bestGain = -0.5;
    
    // precompute the total sum of this node
    // total sum should not change in the scope of this function
    // must be done BEFORE __makeLeaf, __calcMSE and __calcGain
    this->calcEmbedSum(
        y,
        embedM,
        weights,
        indices,
        begin,
        end,
        cache.tSum,
        cache.tW,
        cache
    );

    // first check if hard stop criterion reached
    if (((end - begin) < (2 * this->minSamplesLeaf)) || 
        ((end - begin) < this->minSamplesSplit) ||
        (currentDepth >= maxDepth)) {
        this->__makeLeaf(nodeIdx, y, embedM, indices, begin, end, cache);
        return;
    }

    // then check if node is already pure
    double error;
    error = this->__calcMSE(y, embedM, weights, indices, begin, end, cache);
    if (error < this->stopCriterion) {
        this->__makeLeaf(nodeIdx, y, embedM, indices, begin, end, cache);
        return;
    }

    // samples features to look at
    int featSize = std::min((int)X.n_cols, this->maxFeatures);
    size_t fj = X.n_cols;
    size_t visitedFeats = 0; 

    while ((visitedFeats < featSize) && (fj > nConstants)) {
        visitedFeats++;
        std::uniform_int_distribution<size_t> dist(0, fj - 1);
        size_t currFeatIdx = dist(this->generator);

        if (currFeatIdx < nConstants) continue; //it's a known constant feature
        
        size_t bestFeatSplit;
        double bestFeatGain = -1.0;
        __searchThreshold(
            X,
            y,
            embedM,
            weights,
            indices,
            begin,
            end,
            features, 
            currFeatIdx,
            bestFeatSplit,
            bestFeatGain,
            nConstants,
            cache
        );
        //if (bestFeatGain == -1.0) continue;  // no best split found
        if (bestFeatGain > bestGain) {
            bestGain = bestFeatGain;
            bestFeat = features[currFeatIdx];
            bestSplitPos = bestFeatSplit;
            size_t left = bestFeatSplit - 1;
            size_t right = bestFeatSplit;
            bestThreshold = (cache.Xf(left) + cache.Xf(right))/2.0;
        }
        fj--;
        std::swap(features[fj], features[currFeatIdx]);
    } 
    // check if a good feature & split has been found
    if (bestFeat == -1) {
        // no valid split has been found, then make this node a leaf
        this->__makeLeaf(nodeIdx, y, embedM, indices, begin, end, cache);
        return; 
    }

    // resort by the best feature
    for (int i = begin; i < end; i++) cache.Xf(i) = X(indices(i), bestFeat);
    sort(cache.Xf.memptr() + begin, indices.memptr() + begin, end - begin);

    this->nodesVec[nodeIdx].featureIdx = bestFeat;
    this->nodesVec[nodeIdx].threshold = bestThreshold;
    this->nodesVec[nodeIdx].leftChildIdx = this->nodesVec.size();
    this->nodesVec.emplace_back();
    buildersStack.emplace(
        this->nodesVec[nodeIdx].leftChildIdx,
        begin,
        bestSplitPos,
        currentDepth+1,
        nConstants
    );
    this->nodesVec[nodeIdx].rightChildIdx = this->nodesVec.size();
    this->nodesVec.emplace_back();
    buildersStack.emplace(
        this->nodesVec[nodeIdx].rightChildIdx,
        bestSplitPos,
        end,
        currentDepth+1,
        nConstants
    );
}

template <typename DT, typename ET>
template <typename Ele>
inline void DTRegressor<DT, ET>::_buildNode(
    const nodeBuilder& builder,
    const arma::SpMat<Ele>& X,
    const arma::uvec& y,
    const ET& embedM,
    const arma::vec& weights,
    arma::uvec& indices,
    std::stack<nodeBuilder>& buildersStack, 
    std::vector<size_t>& features,
    treeTrainCache<arma::fmat, ET>& cache
) {
    const int nodeIdx = builder.nodeIdx;
    const size_t begin = builder.begin;
    const size_t end = builder.end;
    const size_t currentDepth = builder.currentDepth;
    size_t nConstants = builder.nConstants;

    int bestSplitPos = -1;
    int bestFeat = -1;
    double bestThreshold = -1; 
    double bestGain = -0.5;

    // precompute the total sum of this node
    // total sum should not change in the scope of this function
    // must be done BEFORE __makeLeaf, __calcMSE and __calcGain
    this->calcEmbedSum(
        y,
        embedM,
        weights,
        indices,
        begin,
        end,
        cache.tSum,
        cache.tW,
        cache
    );

    // first check if hard stop criterion reached
    if (((end - begin) < (2 * this->minSamplesLeaf)) || 
        ((end - begin) < this->minSamplesSplit) ||
        (currentDepth >= maxDepth)) {
        this->__makeLeaf(nodeIdx, y, embedM, indices, begin, end, cache);
        return;
    }

    // then check if node is already pure
    double error;
    error = this->__calcMSE(y, embedM, weights, indices, begin, end, cache);
    if (error < this->stopCriterion) {
        this->__makeLeaf(nodeIdx, y, embedM, indices, begin, end, cache);
        return;
    }

    // samples features to look at
    int featSize = std::min((int)X.n_cols, this->maxFeatures);
    size_t fj = X.n_cols;
    size_t visitedFeats = 0; 
    int isSorted = 0;

    while ((visitedFeats < featSize) && (fj > nConstants)) {
        visitedFeats++;
        std::uniform_int_distribution<size_t> dist(0, fj - 1);
        size_t currFeatIdx = dist(this->generator);

        if (currFeatIdx < nConstants) continue; //it's a known constant feature
        
        size_t bestFeatSplit;
        double bestFeatGain = -1.0;
        double bestFeatThreshold = -1.0;
        __searchThreshold(
            X,
            y,
            embedM,
            weights,
            indices,
            begin,
            end,
            features, 
            currFeatIdx,
            bestFeatSplit,
            bestFeatGain,
            bestFeatThreshold,
            nConstants,
            isSorted,
            cache
        );
        //if (bestFeatGain == -1.0) continue;  // no best split found
        if (bestFeatGain > bestGain) {
            bestGain = bestFeatGain;
            bestFeat = features[currFeatIdx];
            bestSplitPos = bestFeatSplit;
            bestThreshold = bestFeatThreshold;
        }
        fj--;
        std::swap(features[fj], features[currFeatIdx]);
    } 
    // check if a good feature & split has been found
    if (bestFeat == -1) {
        // no valid split has been found, then make this node a leaf
        this->__makeLeaf(nodeIdx, y, embedM, indices, begin, end, cache);
        return; 
    }

    // resort by the best feature
    size_t negEnd, posBegin;
    extractNNZ(
        X,
        indices,
        begin,
        end,
        bestFeat,
        negEnd,
        posBegin,
        isSorted,
        cache
    );
    sort(cache.Xf.memptr() + begin, indices.memptr() + begin, negEnd - begin);
    if (posBegin < end) 
        sort(
            cache.Xf.memptr() + posBegin,
            indices.memptr() + posBegin,
            end - posBegin
        );
    // update the inverse map
    for (size_t i = begin; i < negEnd; i++) cache.row2Idx(indices(i)) = i;
    for (size_t i = posBegin; i < end; i++) cache.row2Idx(indices(i)) = i;

    this->nodesVec[nodeIdx].featureIdx = bestFeat;
    this->nodesVec[nodeIdx].threshold = bestThreshold;
    this->nodesVec[nodeIdx].leftChildIdx = this->nodesVec.size();
    this->nodesVec.emplace_back();
    buildersStack.emplace(
        this->nodesVec[nodeIdx].leftChildIdx,
        begin,
        bestSplitPos,
        currentDepth+1,
        nConstants
    );
    this->nodesVec[nodeIdx].rightChildIdx = this->nodesVec.size();
    this->nodesVec.emplace_back();
    buildersStack.emplace(
        this->nodesVec[nodeIdx].rightChildIdx,
        bestSplitPos,
        end,
        currentDepth+1,
        nConstants
    );
}

template <typename DT, typename ET>
inline void DTRegressor<DT, ET>::train(
    const DT& X,
    const arma::uvec& y,
    const ET& embedM
) {
    arma::vec weights = arma::vec(X.n_rows, arma::fill::ones);
    this->train(X, y, embedM, weights);
}
 
template <typename DT, typename ET>
inline void DTRegressor<DT, ET>::train(
    const DT& X,
    const arma::uvec& y,
    const ET& embedM,
    const arma::vec& weights,
    const unsigned seed
) {
    //fix seed
    generator.seed(seed);
    this->train(X, y, embedM, weights);
}

template <typename DT, typename ET>
inline void DTRegressor<DT, ET>::train(
    const DT& X,
    const arma::uvec& y,
    const ET& embedM,
    const arma::vec& weights
) {
    this->nodesVec.emplace_back();  // emplace the root node

    int nData = X.n_rows;
    arma::uvec indices(nData);
    for (size_t i = 0; i < nData; i++) indices(i) = i;
    
    int nFeatures = X.n_cols;
    std::vector<size_t> features(nFeatures);
    for (size_t i = 0; i < nFeatures; i++) features[i] = i;

    int nEmbed = embedM.n_rows;
    int nClasses = embedM.n_cols;
    // frequently accessed variables to cache:
    treeTrainCache<arma::fmat, ET> cache;
    cache.Xf = arma::fmat(nData, 1);
    cache.lSum = ET(nEmbed, 1);
    cache.rSum = ET(nEmbed, 1);
    cache.tSum = ET(nEmbed, 1);
    cache.row2Idx = arma::uvec(indices);
    cache.sortedIndices = arma::uvec(nData);
    cache.labelCounter = arma::vec(nClasses);
    cache.embedMSq = arma::square(embedM);
 
    // tree construction loop
    std::stack<nodeBuilder> buildersStack;
    buildersStack.emplace(0, 0, nData, 1, 0);
    while(!buildersStack.empty()){
        nodeBuilder currBuilder = buildersStack.top();
        buildersStack.pop();
        _buildNode(
            currBuilder,
            X,
            y,
            embedM,
            weights,
            indices,
            buildersStack,
            features,
            cache
        );
    }
}

template <typename DT, typename ET>
inline void DTRegressor<DT, ET>::predict(const DT& X, ET& preds) {
    DEBUG_MSG("tree inference starts");
    int nData = X.n_rows;
    DEBUG_MSG("done read number of data");
    int targetDim = this->predsVec[0].size();
    DEBUG_MSG("done read embed dim");
    preds = ET(nData, targetDim);
    DEBUG_MSG("loop over test data points");
    for (int i = 0; i < nData; i++) {
        DEBUG_MSG("\n\n***\npredicting the " << i << "th test datapoint");
        size_t nodeIdx = 0;
        while (this->nodesVec[nodeIdx].predIdx == -1) {
            DEBUG_MSG("looking at node " << nodeIdx);
            DEBUG_MSG(
                "splits at "
                << this->nodesVec[nodeIdx].featureIdx
                << "with value "
                << X(i, this->nodesVec[nodeIdx].featureIdx)
                << " with threshold "
                << this->nodesVec[nodeIdx].threshold
                << " children: "
                << this->nodesVec[nodeIdx].leftChildIdx
                << " "
                << this->nodesVec[nodeIdx].rightChildIdx
            );
            if (
                X(i, this->nodesVec[nodeIdx].featureIdx) 
                < this->nodesVec[nodeIdx].threshold
            ) 
                nodeIdx = this->nodesVec[nodeIdx].leftChildIdx;
            else nodeIdx = this->nodesVec[nodeIdx].rightChildIdx;
        }
        DEBUG_MSG("leaf found at node " << nodeIdx);
        preds.row(i) = arma::conv_to<ET>::from(
            this->predsVec[this->nodesVec[nodeIdx].predIdx]
        ).t();
        DEBUG_MSG("done predicting the " << i << "th test datapoint\n***\n\n");
    }
}

template <typename DT, typename ET>
inline void DTRegressor<DT, ET>::printTree(){
    for (size_t i=0; i<this->nodesVec.size(); i++) {
        const DTNode& node = this->nodesVec[i];
        std::cout << "node " << i;
        std::cout << " feature " << node.featureIdx;
        std::cout << " threshold " << node.threshold;
        std::cout << " leftChild " << node.leftChildIdx;
        std::cout << " rightChild " << node.rightChildIdx;
        std::cout << " isLeaf " << (node.predIdx != -1);
        if (node.predIdx != -1) {
            std::cout << " pred: "; 
            std::cout << arma::conv_to<ET>::from(
                this->predsVec[node.predIdx]
            ).t(); 
        }
        std::cout << std::endl;
    } 
}
