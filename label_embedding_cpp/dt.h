#include <algorithm>
#include "embed.h"
#include <time.h>
#include "tree.h"

template <typename DATATYPE, typename EMBEDTYPE>
class DTWithEmbedding: public EmbedBase <DATATYPE, EMBEDTYPE> {
    public:

    DTWithEmbedding(
        int minSamplesSplit_,
        int minSamplesLeaf_,
        int maxFeatures_,
        int maxDepth_,
        double stopCriterion_,
        unsigned seed
    ):
        EmbedBase<DATATYPE, EMBEDTYPE>(),
        tree(
            minSamplesSplit_,
            minSamplesLeaf_,
            maxFeatures_,
            maxDepth_,
            stopCriterion_,
            seed
        ) 
    {}

    DTWithEmbedding(
        int minSamplesSplit_,
        int minSamplesLeaf_,
        int maxFeatures_,
        int maxDepth_,
        double stopCriterion_
    ):
        EmbedBase<DATATYPE, EMBEDTYPE>(),
        tree(
            minSamplesSplit_,
            minSamplesLeaf_,
            maxFeatures_,
            maxDepth_,
            stopCriterion_
        ) 
    {}

    void train(
        const DATATYPE& X,
        const arma::icolvec& y,
        const int embedDim,
        const std::string embedType,
        const arma::icolvec& sortedFlocs_
    );

    double computeRegressionError(
        const DATATYPE& X,
        const arma::icolvec& y
    );

    void predict(const DATATYPE& X, arma::icolvec& preds);

    private:
    DTRegressor<DATATYPE, EMBEDTYPE> tree; 
};

template <typename DT, typename ET>
inline void DTWithEmbedding<DT, ET>::train(
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
    tree.train(X, yIdx, this->embed_m);
}

template <typename DT, typename ET>
inline double DTWithEmbedding<DT, ET>::computeRegressionError(
    const DT& X,
    const arma::icolvec& y
) {
    ET yEmbed;
    this->ConstructEmbeddedTarget(y, yEmbed);  
    ET regPreds;
    tree.predict(X, regPreds);
    return arma::accu(arma::pow(regPreds - yEmbed, 2)) / regPreds.n_rows; 
}

template <typename DT, typename ET>
inline void DTWithEmbedding<DT, ET>::predict(
    const DT& X,
    arma::icolvec& preds
) {
    ET regPreds;
    tree.predict(X, regPreds);
    this->PredictHelper(regPreds, preds);
}
