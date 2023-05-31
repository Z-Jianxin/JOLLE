#include <algorithm>
#include <armadillo>
#include <math.h>

// define the min increment of featur value
#define EXTRACT_NNZ_SWITCH 0.1

// structs
struct DTNode {
    // index to prediction; -1 if not a leaf node.
    int predIdx = -1;
    
    // index of feature used to split; -1 if it is a leaf node.
    int featureIdx = -1;  // this indicates a leaf node
    
    // int thresholdType;  //0 is >; 1 is equal //TODO: consider this later
    
    // threshold to make feature split.
    // TODO: consider more types of splits; like equal to 0 for sparse data
    float threshold = 0;

    // indices of children; -1 if it is a leaf node
    int leftChildIdx = -1;
    int rightChildIdx = -1;
};

template <typename DATATYPE, typename EMBEDTYPE>
struct treeTrainCache{
    DATATYPE Xf;
    EMBEDTYPE lSum;
    EMBEDTYPE rSum;
    EMBEDTYPE tSum;
    arma::uvec row2Idx;  // inverse of indices, used for sparse matrix
    arma::uvec sortedIndices;  // sorted indices between begin and end
    arma::vec labelCounter;
    EMBEDTYPE embedMSq;
    double lW;
    double rW;
    double tW;
};

struct nodeBuilder {
    int nodeIdx = -1;
    size_t begin = 0;
    size_t end = 0;
    size_t currentDepth = 0;
    size_t nConstants = 0;
    nodeBuilder(
        int nodeIdx_,
        size_t begin_,
        size_t end_,
        size_t currentDepth_,
        size_t nConstants_
    ):
        nodeIdx(nodeIdx_),
        begin(begin_),
        end(end_),
        currentDepth(currentDepth_),
        nConstants(nConstants_)
    {}
};

// declarations of sort functions
template<typename Ele, typename Idx>
inline void sort(Ele* Xf, Idx* indices, size_t N);

template<typename Ele, typename Idx>
inline void swap(Ele* Xf, Idx* indices, size_t i, size_t j);

template<typename Ele>
inline Ele median3(Ele* Xf, size_t N);

template<typename Ele, typename Idx>
inline void introsort(Ele* Xf, Idx* indices, size_t N, int maxD);

template<typename Ele, typename Idx>
inline void siftDown(Ele* Xf, Idx* indices, size_t start, size_t end);

template<typename Ele, typename Idx>
inline void heapsort(Ele* Xf, Idx* indices, size_t N);

// implementations of sort functions
// translated from sklearn's Cython implementation
template<typename Ele, typename Idx>
inline void sort(Ele* Xf, Idx* indices, size_t N) {
    if (N == 0) return;
    int maxD = 2 * (int) (log(N));
    introsort(Xf, indices, N, maxD);
}

template<typename Ele, typename Idx>
inline void swap(Ele* Xf, Idx* indices, size_t i, size_t j) {
    std::swap(Xf[i], Xf[j]);
    std::swap(indices[i], indices[j]);
}

template<typename Ele>
inline Ele median3(Ele* Xf, size_t N) {
    Ele a = Xf[0];
    Ele b = Xf[N/2];
    Ele c = Xf[N-1];
    if (a < b) {
        if (b < c) return b;
        else if (a < c) return c;
        else return a;
    } else if (b < c) {
        if (a < c) return a;
        else return c;
    }
    return b;
}

template<typename Ele, typename Idx>
inline void introsort(Ele* Xf, Idx* indices, size_t N, int maxD){
    size_t pivot;
    size_t i, l, r;
    while (N > 1) {
        if (maxD <= 0) {
            heapsort(Xf, indices, N);
            return;
        }
        maxD -= 1;
        pivot = median3(Xf, N);
        i = l = 0;
        r = N;
        while (i < r) {
            if (Xf[i] < pivot) swap(Xf, indices, i++, l++);
            else if (Xf[i] > pivot) swap(Xf, indices, i, --r);
            else i++; 
        }
        introsort(Xf, indices, l, maxD);
        Xf += r;
        indices += r;
        N -= r;
    }
}

template<typename Ele, typename Idx>
inline void siftDown(Ele* Xf, Idx* indices, size_t start, size_t end){
    size_t child, maxInd, root;
    root = start;
    while (1) {
        child = root * 2 + 1;
        maxInd = root;
        if ((child < end) && (Xf[maxInd] < Xf[child])) maxInd = child;
        if (((child+1) < end) && (Xf[maxInd] < Xf[child+1])) maxInd = child+1;
        if (maxInd == root) break;
        else {
            swap(Xf, indices, root, maxInd);
            root = maxInd;
        }
    }
}

template<typename Ele, typename Idx>
inline void heapsort(Ele* Xf, Idx* indices, size_t N) {
    size_t start, end;
    start = (N - 2) / 2;
    end = N;
    while (1) {
        siftDown(Xf, indices, start, end);
        if (start == 0) break;
        start--;
    }
    end = N - 1;
    while (end > 0) {
        swap(Xf, indices, 0, end);
        siftDown(Xf, indices, 0, end);
        end--;
    }
}

// declaratons of extract nzz functions
inline void sparseSwap(
    arma::uvec& indices,
    arma::uvec& row2Idx,
    size_t pos1,
    size_t pos2
);

template <typename Ele, typename EMBEDTYPE>
inline void extractNNZBinary(
    const arma::SpMat<Ele>& X,
    arma::uvec& indices,
    const size_t begin,
    const size_t end,
    const size_t currFeat,
    size_t& negEnd,
    size_t& posBegin,
    int& isSorted,
    treeTrainCache<arma::fmat, EMBEDTYPE>& cache
);

template <typename Ele, typename EMBEDTYPE>
inline void extractNNZInverseIndices(
    const arma::SpMat<Ele>& X,
    arma::uvec& indices,
    const size_t begin,
    const size_t end,
    const size_t currFeat,
    size_t& negEnd,
    size_t& posBegin,
    treeTrainCache<arma::fmat, EMBEDTYPE>& cache
);

template <typename Ele, typename EMBEDTYPE>
inline void extractNNZ(
    const arma::SpMat<Ele>& X,
    arma::uvec& indices,
    const size_t begin,
    const size_t end,
    const size_t currFeat,
    size_t& negEnd,
    size_t& posBegin,
    int& isSorted,
    treeTrainCache<arma::fmat, EMBEDTYPE>& cache
);

// implementations of extract nnz functions
// translated from sklearn's Cython implementation
inline void sparseSwap(
    arma::uvec& indices,
    arma::uvec& row2Idx,
    size_t pos1,
    size_t pos2
) {
    std::swap(indices[pos1], indices[pos2]);
    row2Idx[indices[pos1]] = pos1;
    row2Idx[indices[pos2]] = pos2;
}

template <typename Ele, typename ET>
inline void extractNNZBinary(
    const arma::SpMat<Ele>& X,
    arma::uvec& indices,
    const size_t begin,
    const size_t end,
    const size_t currFeat,
    size_t& negEnd,
    size_t& posBegin,
    int& isSorted,  // 0: unsorted, 1: sorted
    treeTrainCache<arma::fmat, ET>& cache
) {
    if (isSorted == 0) {
        std::copy(
            indices.memptr() + begin,
            indices.memptr() + end,
            cache.sortedIndices.memptr() + begin
        );
        std::sort(
            cache.sortedIndices.memptr() + begin,
            cache.sortedIndices.memptr() + end
        );
        isSorted = 1;
    }
    
    const arma::uword* beginRowPtr = X.row_indices + X.col_ptrs[currFeat];
    const arma::uword* endRowPtr = X.row_indices + X.col_ptrs[currFeat+1];
    while (
        (beginRowPtr < endRowPtr) 
        && (*beginRowPtr < cache.sortedIndices(begin))
    ) ++beginRowPtr;

    while (
        (beginRowPtr < endRowPtr) 
        && (*(endRowPtr-1) > cache.sortedIndices(end-1))
    ) --endRowPtr;

    size_t pos = begin;
    bool found;
    negEnd = begin;
    posBegin = end; 
    while((pos < end) && (beginRowPtr < endRowPtr)) {
        size_t row = cache.sortedIndices(pos);
        found = std::binary_search(beginRowPtr, endRowPtr, row);
        if (!found) {
            pos++;
            continue;
        }
        size_t rowPos = cache.row2Idx(row);
        Ele fVal = X(row, currFeat);
        if (fVal > 0) {
            posBegin--;
            cache.Xf(posBegin) = fVal;
            sparseSwap(indices, cache.row2Idx, rowPos, posBegin);
        }
        else if (fVal < 0) {
            cache.Xf(negEnd) = fVal;
            sparseSwap(indices, cache.row2Idx, rowPos, negEnd);
            negEnd++;
        }
        pos++; 
    }
}

template <typename Ele, typename ET>
inline void extractNNZInverseIndices(
    const arma::SpMat<Ele>& X,
    arma::uvec& indices,
    const size_t begin,
    const size_t end,
    const size_t currFeat,
    size_t& negEnd,
    size_t& posBegin,
    treeTrainCache<arma::fmat, ET>& cache
) {
    const arma::uword* beginRowPtr = X.row_indices + X.col_ptrs[currFeat];
    const arma::uword* endRowPtr = X.row_indices + X.col_ptrs[currFeat+1];
    negEnd = begin;
    posBegin = end; 

    for (auto rowPtr = beginRowPtr; rowPtr < endRowPtr; rowPtr++) {
        size_t rowPos = cache.row2Idx(*rowPtr);
        if ((rowPos < begin) || (rowPos >= end)) continue;
        Ele fVal = X(*rowPtr, currFeat);
        if (fVal > 0) {
            posBegin--;
            cache.Xf(posBegin) = fVal;
            sparseSwap(indices, cache.row2Idx, rowPos, posBegin);
        }
        else if (fVal < 0) {
            cache.Xf(negEnd) = fVal;
            sparseSwap(indices, cache.row2Idx, rowPos, negEnd);
            negEnd++;
        }
    }
}

template <typename Ele, typename ET>
inline void extractNNZ(
    const arma::SpMat<Ele>& X,
    arma::uvec& indices,
    const size_t begin,
    const size_t end,
    const size_t currFeat,
    size_t& negEnd,
    size_t& posBegin,
    int& isSorted,
    treeTrainCache<arma::fmat, ET>& cache
) {
    size_t numData = end - begin;
    size_t nnzFeat = X.col_ptrs[currFeat+1] - X.col_ptrs[currFeat];
    if (
        ((1-isSorted)*numData*log(numData) + numData*log(nnzFeat))
        < (EXTRACT_NNZ_SWITCH * nnzFeat)
    ) {
        extractNNZBinary(
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
        return;    
    }
    extractNNZInverseIndices(
        X,
        indices,
        begin,
        end,
        currFeat,
        negEnd,
        posBegin,
        cache
    );
}

// helper functions for sparse matrix
inline bool sparseIsZero(
    const size_t loc,
    const size_t negEnd,
    const size_t posBegin
) {return ((loc >= negEnd) && (loc < posBegin));}

template <typename ET>
inline float sparseGet(
    const size_t loc,
    const size_t negEnd,
    const size_t posBegin,
    const treeTrainCache<arma::fmat, ET>& cache
) {
    if(sparseIsZero(loc, negEnd, posBegin)) return 0.0;
    return cache.Xf(loc); 
}

inline void sparseIncrement(
    size_t& loc,
    size_t& prevLoc,
    const size_t negEnd,
    const size_t posBegin
) {
    if ((loc + 1) == negEnd) { 
        loc = posBegin;
        prevLoc = negEnd - 1;
        return;
    }
    loc++;
    prevLoc = loc - 1;
}

