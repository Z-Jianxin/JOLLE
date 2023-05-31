#include <armadillo>
#include <chrono>
#include "embed.h"
#include <iostream>
#include <math.h>
#include <vector>

#define ZERO_TOL 1e-7

template <typename DATATYPE, typename PARAMTYPE, typename EMBEDTYPE>
class pdenParaWithEmbedding: public EmbedBase <DATATYPE, EMBEDTYPE>{
    public:
    pdenParaWithEmbedding():
        EmbedBase<DATATYPE, EMBEDTYPE>(),
        lambda1(0.1),
        lambda2(0.5)
    {}

    pdenParaWithEmbedding(double lambda1_, double lambda2_):
        EmbedBase<DATATYPE, EMBEDTYPE>(),
        lambda1(lambda1_),
        lambda2(lambda2_)
    {}

    template<typename EleData>
    void train(
        const arma::SpMat<EleData>& X,
        const arma::icolvec& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const std::string embedType,
        const arma::icolvec& sortedFlocs_
    );

    template<typename EleData>
    void train(
        const arma::SpMat<EleData>& X,
        const arma::icolvec& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const unsigned startDim,
        const unsigned endDim,
        const std::string embedType,
        const arma::icolvec& sortedFlocs_,
        const int seed
    );

    
    template<typename EleData, typename EleDual>
    void train(
        const arma::SpMat<EleData>& X,
        const arma::icolvec& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const unsigned startDim,
        const unsigned endDim,
        const std::string embedType,
        const arma::icolvec& sortedFlocs_,
        EleDual* V,
        EleDual* XtrV
    );

    double computeDualityGap(
        const DATATYPE& X,
        const arma::icolvec& y,
        const float* const V,
        const unsigned nData,
        const unsigned nEmbed
    );

    double computeRegressionError(
        const DATATYPE& X, 
        const arma::icolvec& y
    );
 
    void predict(const DATATYPE& X, arma::icolvec& preds);
    PARAMTYPE weights; // gather trained model for MPI

    private:
    double lambda1; //parameter for l1 regularization
    double lambda2; //parameter for l2 regularization
    EMBEDTYPE Vtr;  // transpose of dual variable
};

template <typename DATATYPE, typename PARAMTYPE, typename EMBEDTYPE>
class pdenMultiLabel: public EmbedMultiBase <DATATYPE, EMBEDTYPE>{
    public:
    pdenMultiLabel(unsigned K, double lambda1_, double lambda2_):
        EmbedMultiBase<DATATYPE, EMBEDTYPE>(K),
        // K = max size of eta(x)'s support
        lambda1(lambda1_),
        lambda2(lambda2_)
    {}

    template<typename EleData>
    void train(
        const arma::SpMat<EleData>& X,
        const std::vector<std::vector<unsigned>>& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const std::string embedType,
        const unsigned nClasses
    );

    template<typename EleData>
    void train(
        const arma::SpMat<EleData>& X,
        const std::vector<std::vector<unsigned>>& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const unsigned startDim,
        const unsigned endDim,
        const std::string embedType,
        const int seed,
        const unsigned nClasses
    );

    
    template<typename EleData, typename EleDual>
    void train(
        const arma::SpMat<EleData>& X,
        const std::vector<std::vector<unsigned>>& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const unsigned startDim,
        const unsigned endDim,
        const std::string embedType,
        EleDual* V,
        EleDual* XtrV,
        const unsigned nClasses
    );

    double computeDualityGap(
        const DATATYPE& X,
        const std::vector<std::vector<unsigned>>& y,
        const float* const V,
        const unsigned nData,
        const unsigned nEmbed
    );

    double computeRegressionError(
        const DATATYPE& X, 
        const std::vector<std::vector<unsigned>>& y
    );
 
    void predict_omp(
        const DATATYPE& X,
        std::vector<std::vector<unsigned>>& preds,
        const float thresh=0.5
    );

    void topk_omp(
        const DATATYPE& X,
        std::vector<std::vector<unsigned>>& preds,
        const unsigned k
    );

    void predict_probs_omp(
        const DATATYPE& X,  //use std::move if X no longer needed
        unsigned long long * labels,
        float * probs
    );
 
    PARAMTYPE weights; // gather trained model for MPI

    private:
    double lambda1; //parameter for l1 regularization
    double lambda2; //parameter for l2 regularization
};

template <typename EleData, typename IDX, typename EleModel, typename EleEmbed>
inline void update(
    const EleData* const XtrValues,
    const IDX* const XtrRowIndices,
    const unsigned start,
    const unsigned end,
    const EleEmbed embedk,
    const EleData Qi,
    EleModel& Vik,
    EleModel* XtrVk,
    EleModel* Wk,
    const double lambda1,
    const double lambda2
) {
    /*update the dual*/
    double inner = 0.0;
    for (unsigned i = start; i < end; i++) {
        IDX rowIdx = XtrRowIndices[i];
        inner += XtrValues[i] * Wk[rowIdx];
    }
    double currGrad = 0.5 * Vik - embedk + inner;
    currGrad /= (Qi / 2.0);
    Vik -= currGrad;
    /*update the dual done*/
    /*maintain the primal*/
    double twoLambda2 = 2 * lambda2;
    double minusLambda1 = -lambda1;
    for (unsigned i = start; i < end; i++) {
        IDX rowIdx = XtrRowIndices[i];
        XtrVk[rowIdx] -= currGrad * XtrValues[i];
        if (lambda1 < ZERO_TOL) {
            Wk[rowIdx] = XtrVk[rowIdx] / twoLambda2;
            continue;
        }
        if (XtrVk[rowIdx] > lambda1) 
            Wk[rowIdx] = (XtrVk[rowIdx] - lambda1) / twoLambda2;
        else if (XtrVk[rowIdx] < minusLambda1) 
            Wk[rowIdx] = (XtrVk[rowIdx] + lambda1) / twoLambda2;
        else Wk[rowIdx]= 0.0;
    }
    /*maintain the primal done*/
}

template <typename EleData, typename IDX, typename EleModel, typename EleEmbed>
inline void postUpdate(
    const EleData* const XtrValues,
    const IDX* const XtrRowIndices,
    const unsigned start,
    const unsigned end,
    const EleEmbed embedk,
    const EleData Qi,
    EleModel& Vik,
    EleModel* XtrVk,
    EleModel* Wk,
    const double lambda1,
    const double lambda2,
    const EleModel& Vik_,
    const EleModel* const Wk_
) {
    /*update the dual*/
    double inner = 0.0;
    if (fabs(Vik_) >= ZERO_TOL) {
        for (unsigned i = start; i < end; i++) {
            IDX rowIdx = XtrRowIndices[i];
            if (fabs(Wk_[rowIdx]) < ZERO_TOL) continue;
            inner += XtrValues[i] * Wk[rowIdx];
        }
    }
    double currGrad = 0.5 * Vik - embedk + inner;
    currGrad /= (Qi/ 2.0);
    Vik -= currGrad;
    if (fabs(Vik_) < ZERO_TOL) return;  
    /*update the dual done*/
    /*maintain the primal*/
    double twoLambda2 = 2 * lambda2;
    double minusLambda1 = -lambda1;
    for (unsigned i = start; i < end; i++) {
        IDX rowIdx = XtrRowIndices[i];
        if (fabs(Wk_[rowIdx]) < ZERO_TOL) continue;
        Wk[rowIdx] -= (currGrad * XtrValues[i]) / twoLambda2;
    }
    /*maintain the primal done*/
}

template <typename EleData, typename IDX, typename EleModel, typename EleEmbed>
inline void workerSolve(
    const EleData* const XtrValues,
    const IDX* const XtrRowIndices,
    const IDX* const XtrColPtrs,
    //const EleEmbed* const embed,
    const EleEmbed* const target,
    const EleData* const Q,
    //const IDX* const yIdx,
    EleModel* Vk,
    EleModel* XtrVk,
    EleModel* Wk,
    //const unsigned outIdx,
    const double lambda1,
    const double lambda2,
    const unsigned nData,
    const unsigned nFeat,
    //const unsigned nEmbed,
    const unsigned nIter,
    const unsigned postIter
) {
    for (unsigned t = 0; t < nIter; ++t) {
        arma::uvec samples = arma::randperm(nData);
        for (unsigned j = 0; j < nData; j++) {
            unsigned i = samples(j);
            unsigned start = XtrColPtrs[i];
            unsigned end = XtrColPtrs[i+1];
            //EleEmbed embedk = embed[yIdx[i] * nEmbed + outIdx];
            EleEmbed embedk = target[i];
            update(
                XtrValues,
                XtrRowIndices,
                start,
                end,
                embedk,
                Q[i],
                Vk[i],
                XtrVk,
                Wk,
                lambda1,
                lambda2
            );
        }
    }
    if (postIter == 0) return;

    /*construct submatrix for post processing*/
    //auto start = std::chrono::steady_clock::now();

    float* Vk_ = new float[nData];
    float* Wk_ = new float[nFeat];
    memcpy(Vk_, Vk, nData*sizeof(float));
    memcpy(Wk_, Wk, nFeat*sizeof(float)); 
    //arma::frowvec QRecon = arma::sum(arma::square(XtrRecon), 0) / lambda2 + 1;
    arma::frowvec QRecon = arma::frowvec(nData, arma::fill::zeros);
    for (unsigned i = 0; i < nData; ++i) {
        if (fabs(Vk[i]) >= ZERO_TOL) {
            unsigned start = XtrColPtrs[i];
            unsigned end = XtrColPtrs[i+1];
            for (unsigned j = start; j < end; ++j) {
                IDX rowIdx = XtrRowIndices[j];
                if (fabs(Wk[rowIdx]) < ZERO_TOL) continue;
                QRecon(i) += XtrValues[j] * XtrValues[j];
            }
        }
    }
    QRecon = QRecon / lambda2 + 1;
    /*auto end = std::chrono::steady_clock::now();
    std::cout << "post construction time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>
        (end-start).count() / 1e6 << "s";
    std::cout << std::endl;*/

    //start = std::chrono::steady_clock::now();
    for (unsigned t = 0; t < postIter; ++t) {
        arma::uvec samples = arma::randperm(nData);
        for (unsigned j = 0; j < nData; ++j) {
            unsigned i = samples(j);
            unsigned start = XtrColPtrs[i];
            unsigned end = XtrColPtrs[i+1];
            //EleEmbed embedk = embed[yIdx[i] * nEmbed + outIdx];
            EleEmbed embedk = target[i];
            postUpdate(
                XtrValues,
                XtrRowIndices,
                start,
                end,
                embedk,
                QRecon(i),
                Vk[i],
                XtrVk,
                Wk,
                lambda1,
                lambda2,
                Vk_[i],
                Wk_
            );
        }
    }
    /*end = std::chrono::steady_clock::now();
    std::cout << "post update time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>
        (end-start).count() / 1e6 << "s";
    std::cout << std::endl;*/ 
}

template <typename DT, typename PT, typename ET>
template<typename EleData>
inline void pdenParaWithEmbedding<DT, PT, ET>::train(
        const arma::SpMat<EleData>& X,
        const arma::icolvec& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const std::string embedType,
        const arma::icolvec& sortedFlocs_
) {
    // TODO: parameter type as template
    const unsigned nData = X.n_rows;
    const unsigned nFeat = X.n_cols;
    float* V = new float[nData * embedDim];
    float* XtrV = new float[nFeat * embedDim];
    auto start = std::chrono::steady_clock::now();
    this->train(
        X,
        y,
        nIter,
        nPostIter,
        embedDim,
        0,
        embedDim,
        embedType,
        sortedFlocs_,
        V,
        XtrV
    ); 
    auto end = std::chrono::steady_clock::now();
    std::cout << "Training completes." << std::endl;
    std::cout << "Training time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>
                 (end-start).count() / 1e6 << "s";
    std::cout << std::endl;

    double gap = this->computeDualityGap(X, y, V, nData, embedDim);
    std::cout << "duality gap = " << gap << std::endl;
    delete V;
    delete XtrV; 
}

template <typename DT, typename PT, typename ET>
template<typename EleData>
inline void pdenParaWithEmbedding<DT, PT, ET>::train(
    const arma::SpMat<EleData>& X,
    const arma::icolvec& y,
    const unsigned nIter,
    const unsigned nPostIter,
    const unsigned embedDim,
    const unsigned startDim,
    const unsigned endDim,
    const std::string embedType,
    const arma::icolvec& sortedFlocs_,
    const int seed
) {
    arma::arma_rng::set_seed(seed);
    const unsigned nData = X.n_rows;
    const unsigned nFeat = X.n_cols;
    float* V = new float[nData * (endDim - startDim)];
    float* XtrV = new float[nFeat * (endDim - startDim)];
    this->train(
        X,
        y,
        nIter,
        nPostIter,
        embedDim,
        startDim,
        endDim,
        embedType,
        sortedFlocs_,
        V,
        XtrV
    ); 

    delete V;
    delete XtrV;
}

template <typename DT, typename PT, typename ET>
template<typename EleData, typename EleDual>
inline void pdenParaWithEmbedding<DT, PT, ET>::train(
    const arma::SpMat<EleData>& X,
    const arma::icolvec& y,
    const unsigned nIter,
    const unsigned nPostIter,
    const unsigned embedDim,
    const unsigned startDim,
    const unsigned endDim,
    const std::string embedType,
    const arma::icolvec& sortedFlocs_,
    EleDual* V,
    EleDual* XtrV
){
    /*initialize embedding matrx and label indices*/
    this->check_input(y, sortedFlocs_);
    int nClasses = this->sorted_labels.n_rows;
    GetEmbeddingMatrix(embedType, nClasses, embedDim, this->embed_m);
    arma::uvec yIdx;
    this->y2yIdx(y, yIdx);

    const unsigned nData = X.n_rows;
    const unsigned nFeat = X.n_cols;
    const unsigned nEmbed = this->embed_m.n_rows;

    DT Xtr = X.t().eval();
    Xtr.sync();

    this->weights.zeros(nFeat, endDim - startDim);
    memset(V, 0, sizeof(EleDual) * nData * (endDim - startDim));
    memset(XtrV, 0, sizeof(EleDual) * nFeat * (endDim - startDim));
    arma::frowvec Q = arma::sum(arma::square(Xtr), 0) / this->lambda2 + 1;

    #pragma omp parallel for
    for (unsigned k = startDim; k < endDim; ++k) {
        float* target = new float[nData];
        memset(target, 0, sizeof(float) * nData);
        for (unsigned i = 0; i < nData; ++i) {
            target[i] = this->embed_m(k, yIdx(i));
        }
        workerSolve(
            Xtr.values,
            Xtr.row_indices,
            Xtr.col_ptrs,
            //this->embed_m.memptr(),
            target,
            Q.memptr(),
            //yIdx.memptr(),
            V + nData * (k - startDim),
            XtrV + nFeat * (k - startDim),
            this->weights.memptr() + nFeat * (k - startDim),
            //k,
            this->lambda1,
            this->lambda2,
            nData,
            nFeat,
            //nEmbed,
            nIter,
            nPostIter
        );
        delete target;
    }
}

template <typename DT, typename PT, typename ET>
inline double pdenParaWithEmbedding<DT, PT, ET>::computeDualityGap(
        const DT& X, 
        const arma::icolvec& y,
        const float* const V,
        const unsigned nData,
        const unsigned nEmbed
) {
    ET yEmbed;
    this->ConstructEmbeddedTarget(y, yEmbed); 
    ET regPreds(X * this->weights);

    double primal = arma::accu(arma::square(regPreds - yEmbed))
        + this->lambda1 * arma::accu(arma::abs(this->weights))
        + this->lambda2 * arma::accu(arma::square(this->weights));

    double dual = 0.0;
    for (unsigned k = 0; k < nEmbed; ++k){
        unsigned base = k * nData;
        for (unsigned i = 0; i < nData; ++i) {
            dual +=  yEmbed(i, k) * V[base + i] - pow(V[base + i], 2) / 4.0;
        }
    }
    dual -= this->lambda2 * arma::accu(arma::square(this->weights));
    return primal - dual;
}

template <typename DT, typename PT, typename ET>
inline double pdenParaWithEmbedding<DT, PT, ET>::computeRegressionError(
        const DT& X, 
        const arma::icolvec& y
){
    ET yEmbed;
    this->ConstructEmbeddedTarget(y, yEmbed); 
    ET regPreds(X * this->weights);
    return arma::accu(arma::pow(regPreds - yEmbed, 2)) / regPreds.n_rows;
}

template <typename DT, typename PT, typename ET>
inline void pdenParaWithEmbedding<DT, PT, ET>::predict(
    const DT& X,  //use std::move if X no longer needed
    arma::icolvec& preds
){
    ET regPreds(X * this->weights);
    this->PredictHelper(regPreds, preds); 
}

template <typename DT, typename PT, typename ET>
template<typename EleData>
inline void pdenMultiLabel<DT, PT, ET>::train(
        const arma::SpMat<EleData>& X,
        const std::vector<std::vector<unsigned>>& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const std::string embedType,
        const unsigned nClasses
) {
    const unsigned nData = X.n_rows;
    const unsigned nFeat = X.n_cols;
    float* V = new float[nData * embedDim];
    float* XtrV = new float[nFeat * embedDim];
    auto start = std::chrono::steady_clock::now();
    this->train(
        X,
        y,
        nIter,
        nPostIter,
        embedDim,
        0,
        embedDim,
        embedType,
        V,
        XtrV,
        nClasses
    ); 
    auto end = std::chrono::steady_clock::now();
    std::cout << "Training completes." << std::endl;
    std::cout << "Training time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>
                 (end-start).count() / 1e6 << "s";
    std::cout << std::endl;

    double gap = this->computeDualityGap(X, y, V, nData, embedDim);
    std::cout << "duality gap = " << gap << std::endl;
    delete V;
    delete XtrV; 
}

template <typename DT, typename PT, typename ET>
template<typename EleData>
inline void pdenMultiLabel<DT, PT, ET>::train(
    const arma::SpMat<EleData>& X,
    const std::vector<std::vector<unsigned>>& y,
    const unsigned nIter,
    const unsigned nPostIter,
    const unsigned embedDim,
    const unsigned startDim,
    const unsigned endDim,
    const std::string embedType,
    const int seed,
    const unsigned nClasses
) {
    arma::arma_rng::set_seed(seed);
    const unsigned nData = X.n_rows;
    const unsigned nFeat = X.n_cols;
    float* V = new float[nData * (endDim - startDim)];
    float* XtrV = new float[nFeat * (endDim - startDim)];
    this->train(
        X,
        y,
        nIter,
        nPostIter,
        embedDim,
        startDim,
        endDim,
        embedType,
        V,
        XtrV,
        nClasses
    ); 

    delete V;
    delete XtrV;
}

template <typename DT, typename PT, typename ET>
template<typename EleData, typename EleDual>
inline void pdenMultiLabel<DT, PT, ET>::train(
    const arma::SpMat<EleData>& X,
    const std::vector<std::vector<unsigned>>& y,
    const unsigned nIter,
    const unsigned nPostIter,
    const unsigned embedDim,
    const unsigned startDim,
    const unsigned endDim,
    const std::string embedType,
    EleDual* V,
    EleDual* XtrV,
    const unsigned nClasses
){
    /*initialize embedding matrx and label indices*/
    //int nClasses = this->embed_m.n_cols;
    GetEmbeddingMatrix(embedType, nClasses, embedDim, this->embed_m);

    const unsigned nData = X.n_rows;
    const unsigned nFeat = X.n_cols;
    const unsigned nEmbed = this->embed_m.n_rows;

    DT Xtr = X.t().eval();
    Xtr.sync();

    this->weights.zeros(nFeat, endDim - startDim);
    memset(V, 0, sizeof(EleDual) * nData * (endDim - startDim));
    memset(XtrV, 0, sizeof(EleDual) * nFeat * (endDim - startDim));
    arma::frowvec Q = arma::sum(arma::square(Xtr), 0) / this->lambda2 + 1;

    #pragma omp parallel for
    for (unsigned k = startDim; k < endDim; ++k) {
        float* target = new float[nData];
        memset(target, 0, sizeof(float) * nData);
        for (unsigned i = 0; i < nData; ++i) {
            for (unsigned label: y[i]) {
                target[i] += this->embed_m(k, label);
            }
        }
        workerSolve(
            Xtr.values,
            Xtr.row_indices,
            Xtr.col_ptrs,
            //this->embed_m.memptr(),
            target,
            Q.memptr(),
            //yIdx.memptr(),
            V + nData * (k - startDim),
            XtrV + nFeat * (k - startDim),
            this->weights.memptr() + nFeat * (k - startDim),
            //k,
            this->lambda1,
            this->lambda2,
            nData,
            nFeat,
            //nEmbed,
            nIter,
            nPostIter
        );
        delete target;
    }
}

template <typename DT, typename PT, typename ET>
inline double pdenMultiLabel<DT, PT, ET>::computeDualityGap(
        const DT& X, 
        const std::vector<std::vector<unsigned>>& y,
        const float* const V,
        const unsigned nData,
        const unsigned nEmbed
) {
    ET yEmbed;
    this->ConstructEmbeddedTarget(y, yEmbed); 
    ET regPreds(X * this->weights);

    double primal = arma::accu(arma::square(regPreds - yEmbed))
        + this->lambda1 * arma::accu(arma::abs(this->weights))
        + this->lambda2 * arma::accu(arma::square(this->weights));

    double dual = 0.0;
    for (unsigned k = 0; k < nEmbed; ++k){
        unsigned base = k * nData;
        for (unsigned i = 0; i < nData; ++i) {
            dual +=  yEmbed(i, k) * V[base + i] - pow(V[base + i], 2) / 4.0;
        }
    }
    dual -= this->lambda2 * arma::accu(arma::square(this->weights));
    return primal - dual;
}

template <typename DT, typename PT, typename ET>
inline double pdenMultiLabel<DT, PT, ET>::computeRegressionError(
        const DT& X, 
        const std::vector<std::vector<unsigned>>& y
){
    ET yEmbed;
    this->ConstructEmbeddedTarget(y, yEmbed); 
    ET regPreds(X * this->weights);
    return arma::accu(arma::pow(regPreds - yEmbed, 2)) / regPreds.n_rows;
}

template <typename DT, typename PT, typename ET>
inline void pdenMultiLabel<DT, PT, ET>::predict_omp(
    const DT& X,  //use std::move if X no longer needed
    std::vector<std::vector<unsigned>>& preds,
    const float thresh
){
    ET regPreds((X * this->weights).t());
    unsigned nEmbed = this->embed_m.n_rows;
    unsigned nClasses = this->embed_m.n_cols;
    unsigned nData = X.n_rows;
    #pragma omp parallel for 
    for (unsigned i=0; i<nData; ++i){
        //arma::uword * cols = new arma::uword[this->K];
        unsigned long long * cols = new unsigned long long [this->K];
        float * yptr = new float[this->K];
        omp(
            this->embed_m.memptr(),
            regPreds.memptr() + nEmbed * i,
            cols,
            yptr,
            nEmbed,
            nClasses,
            this->K
        );
        for (unsigned j=0; j<this->K; ++j) {
            if (yptr[j] >= thresh) preds[i].push_back(cols[j]);
        }
        if (preds[i].size() == 0) 
            preds[i].push_back(
                cols[std::max_element(yptr, yptr+this->K) - yptr]
            );
        delete yptr;
        delete cols;
    }
}

template <typename DT, typename PT, typename ET>
inline void pdenMultiLabel<DT, PT, ET>::predict_probs_omp(
    const DT& X,  //use std::move if X no longer needed
    unsigned long long * labels,
    float * probs
){
    ET regPreds((X * this->weights).t());
    unsigned nEmbed = this->embed_m.n_rows;
    unsigned nClasses = this->embed_m.n_cols;
    unsigned nData = X.n_rows;

    #pragma omp parallel for 
    for (unsigned i=0; i<nData; ++i){
        unsigned long long * cols = new unsigned long long [this->K];
        float * yptr = new float[this->K];
        omp(
            this->embed_m.memptr(),
            regPreds.memptr() + nEmbed * i,
            cols,
            yptr,
            nEmbed,
            nClasses,
            this->K
        );
        memcpy(
            labels + this->K * i,
            cols,
            this->K * sizeof(unsigned long long)
        );
        memcpy(probs + this->K * i, yptr, this->K * sizeof(float));
        delete yptr;
        delete cols;
    }
}

template <typename DT, typename PT, typename ET>
inline void pdenMultiLabel<DT, PT, ET>::topk_omp(
    const DT& X,  //use std::move if X no longer needed
    std::vector<std::vector<unsigned>>& preds,
    const unsigned k
){
    ET regPreds((X * this->weights).t());
    unsigned nEmbed = this->embed_m.n_rows;
    unsigned nClasses = this->embed_m.n_cols;
    unsigned nData = X.n_rows;
    #pragma omp parallel for 
    for (unsigned i=0; i<nData; ++i){
        //arma::uword * cols = new arma::uword[this->K];
        unsigned long long * cols = new unsigned long long [this->K];
        float * yptr = new float[this->K];
        omp(
            this->embed_m.memptr(),
            regPreds.memptr() + nEmbed * i,
            cols,
            yptr,
            nEmbed,
            nClasses,
            this->K
        );
        std::vector<unsigned> indices(this->K);
        for (unsigned t=0; t<this->K; ++t) indices[t]=t;
        std::sort(
            indices.begin(),
            indices.end(),
            [yptr](const unsigned a, const unsigned b) {
                return yptr[a] > yptr[b];
            }  // use > to get descending order
        );
        for (unsigned j=0; j<k; ++j) {
            preds[i].push_back(cols[indices[j]]);
        }
        delete yptr;
        delete cols;
    }
}
