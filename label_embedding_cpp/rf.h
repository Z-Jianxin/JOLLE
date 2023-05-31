#include <chrono>
#include "embed.h"
#include <random>
#include <time.h>
#include "tree.h"
#include <vector>

template <typename DATATYPE, typename EMBEDTYPE>
class RFWithEmbedding: public EmbedBase <DATATYPE, EMBEDTYPE> {
    public:
    RFWithEmbedding(
        size_t nEstimators_,
        int minSamplesSplit_,
        int minSamplesLeaf_,
        int maxFeatures_,
        int maxDepth_,
        double stopCriterion_,
        unsigned seed
    ):
        EmbedBase<DATATYPE, EMBEDTYPE>(),
        nEstimators(nEstimators_),
        treeVec(
            nEstimators_,
            DTRegressor<DATATYPE, EMBEDTYPE>(
                minSamplesSplit_,
                minSamplesLeaf_,
                maxFeatures_,
                maxDepth_,
                stopCriterion_
            )
        ),
        generator(seed) 
    {}

    void train(
        const DATATYPE& X,
        const arma::icolvec& y,
        const int embedDim,
        const std::string embedType,
        const arma::icolvec& sortedFlocs_
    );

    void train(
        const DATATYPE& X,
        const arma::icolvec& y,
        const int embedDim,
        const std::string embedType,
        const arma::icolvec& sortedFlocs_,
        const int embed_seed
    );

    double computeRegressionError(
        const DATATYPE& X,
        const arma::icolvec& y
    );

    void predictEmbed(const DATATYPE& X, EMBEDTYPE& preds);
    void predict(const DATATYPE& X, arma::icolvec& preds);
    void predictVote(const DATATYPE& X, arma::icolvec& preds);
    size_t nEstimators;

    private:
    std::vector<DTRegressor<DATATYPE, EMBEDTYPE>> treeVec; 
    std::default_random_engine generator;
};

template <typename DT, typename ET>
inline void RFWithEmbedding<DT, ET>::train(
    const DT& X,
    const arma::icolvec& y,
    const int embedDim,
    const std::string embedType,
    const arma::icolvec& sortedFlocs_
) {
    this->check_input(y, sortedFlocs_);
    int numClasses = this->sorted_labels.n_rows;
    GetEmbeddingMatrix(embedType, numClasses, embedDim, this->embed_m);
 
    arma::uvec yIdx;
    this->y2yIdx(y, yIdx);
    
    const size_t nData = X.n_rows;
    const auto& embedM = this->embed_m;

    // seeds for bootstrapping
    std::vector<size_t> bsSeedsVec(this->nEstimators);
    // seeds for training trees
    std::vector<size_t> treeSeedsVec(this->nEstimators);

    std::uniform_int_distribution<size_t> sampler;
    for (size_t i = 0; i < this->nEstimators; i++) {
        bsSeedsVec[i] = sampler(this->generator);
        treeSeedsVec[i] = sampler(this->generator);
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < this->treeVec.size(); i++) {
        thread_local std::default_random_engine generator(bsSeedsVec[i]);
        std::uniform_int_distribution<size_t> dist(0, nData - 1);
        arma::vec weights = arma::vec(nData, arma::fill::zeros) + 1e-6;
        for (size_t i = 0; i < nData; i++) weights(dist(generator)) += 1.0;
        this->treeVec[i].train(X, yIdx, embedM, weights, treeSeedsVec[i]);
    }
}

template <typename DT, typename ET>
inline void RFWithEmbedding<DT, ET>::train(
    const DT& X,
    const arma::icolvec& y,
    const int embedDim,
    const std::string embedType,
    const arma::icolvec& sortedFlocs_,
    const int embed_seed
) {
    arma::arma_rng::set_seed(embed_seed);
    this->train(X, y, embedDim, embedType, sortedFlocs_);
}

template <typename DT, typename ET>
inline double RFWithEmbedding<DT, ET>::computeRegressionError(
    const DT& X,
    const arma::icolvec& y
) {
    size_t nEmbed = this->embed_m.n_rows;
    size_t nData = X.n_rows;
    ET yEmbed;
    this->ConstructEmbeddedTarget(y, yEmbed);  
    
    ET regPreds(nData, nEmbed, arma::fill::zeros);
    #pragma omp declare reduction (matrixSum: ET: omp_out += omp_in) \
        initializer (omp_priv=omp_orig)
    #pragma omp parallel for reduction(matrixSum: regPreds)
    for (size_t i = 0; i < this->treeVec.size(); i++) {
        ET treePreds;
        this->treeVec[i].predict(X, treePreds);
        regPreds += treePreds; 
    }
    regPreds /= this->nEstimators; 
    
    return arma::accu(arma::pow(regPreds - yEmbed, 2)) / regPreds.n_rows; 
}

template <typename DT, typename ET>
inline void RFWithEmbedding<DT, ET>::predictEmbed(
    const DT& X,
    ET& preds
) {
    size_t nEmbed = this->embed_m.n_rows;
    size_t nData = X.n_rows;
    preds.zeros(nData, nEmbed);
    #pragma omp declare reduction (matrixSum: ET: omp_out += omp_in) \
        initializer (omp_priv=omp_orig)
    #pragma omp parallel for reduction(matrixSum: preds)
    for (size_t i = 0; i < this->treeVec.size(); i++) {
        ET treePreds;
        this->treeVec[i].predict(X, treePreds);
        preds += treePreds;
    }
    preds /= this->nEstimators; 
}

template <typename DT, typename ET>
inline void RFWithEmbedding<DT, ET>::predict(
    const DT& X,
    arma::icolvec& preds
) {
    size_t nEmbed = this->embed_m.n_rows;
    size_t nData = X.n_rows;
    ET regPreds(nData, nEmbed, arma::fill::zeros);
    #pragma omp declare reduction (matrixSum: ET: omp_out += omp_in) \
        initializer (omp_priv=omp_orig)
    #pragma omp parallel for reduction(matrixSum: regPreds)
    for (size_t i = 0; i < this->treeVec.size(); i++) {
        ET treePreds;
        this->treeVec[i].predict(X, treePreds);
        regPreds += treePreds;
    }
    regPreds /= this->nEstimators; 
    this->PredictHelper(regPreds, preds);
}

template <typename DT, typename ET>
inline void RFWithEmbedding<DT, ET>::predictVote(
    const DT& X,
    arma::icolvec& preds
) {
    size_t nEmbed = this->embed_m.n_rows;
    size_t nClasses = this->embed_m.n_cols;
    size_t nData = X.n_rows;
    arma::umat votes(nData, nClasses, arma::fill::zeros);
    #pragma omp parallel for
    for (size_t i = 0; i < this->treeVec.size(); i++) {
        ET treePredsReg;
        arma::uvec treePredsLabelIdx;
        this->treeVec[i].predict(X, treePredsReg);
        find1NN(
            treePredsReg,
            this->embed_m,
            this->good_label,
            this->sorted_labels,
            treePredsLabelIdx
        );
        for (size_t i = 0; i < nData; i++) votes(i, treePredsLabelIdx(i)) += 1;
    }
    arma::uvec predsIdx = arma::index_max(votes, 1);
    preds = arma::icolvec(predsIdx.n_rows);
    if (this->good_label){
        preds = arma::conv_to<arma::icolvec>::from(std::move(predsIdx));
        return;
    }
    std::transform(
        predsIdx.begin(),
        predsIdx.end(),
        preds.begin(),
        [this](size_t idx) {return this->sorted_labels(idx);}
    );
}
