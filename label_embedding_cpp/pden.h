#include <algorithm>
#include <armadillo>
#include "embed.h"
#include <iostream>
#include <math.h>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(DEBUG)
#define DEBUG_MSG(str) do {std::cerr << "DEBUG: " << str << std::endl;} \
    while (false)
#else
#define DEBUG_MSG(str) do { } while (false)
#endif

#define ZERO_TOL 1e-7

template<typename Ele>
inline unsigned countNNZ(arma::SpMat<Ele> M) {return M.n_nonzero;}

template<typename Ele>
inline unsigned countNNZ(arma::Mat<Ele> M) {return arma::accu(M != 0);}

template <typename EMBEDTYPE, typename Ele1, typename Ele2>
inline unsigned selectIndex(
    const arma::SpMat<Ele1>& Xtr,
    const arma::Mat<Ele2>& W,
    const EMBEDTYPE& embedM,
    const unsigned labelCol,
    const unsigned sampleIdx,
    EMBEDTYPE& predApprox
);

/* select active variable by importance sampling*/
template <typename EMBEDTYPE, typename Ele>
inline unsigned selectIndex(
    const std::vector<float>& sampleAbs,
    const std::vector<int>& samplePos,
    const std::vector<unsigned>& rowIdx,
    const float l1norm,
    const unsigned featSampleNum,
    const arma::Mat<Ele>& W,
    const EMBEDTYPE& embedM,
    const unsigned labelCol,
    EMBEDTYPE& predApprox,
    std::default_random_engine& generator
);

template <typename EMBEDTYPE, typename Ele1, typename Ele2>
inline unsigned selectIndex(
    const arma::SpMat<Ele1>& Xtr,
    const arma::SpMat<Ele2>& W,
    const EMBEDTYPE& embedM,
    const unsigned labelCol,
    const unsigned sampleIdx
);

template <typename Ele1, typename Ele2>
inline double innerProd(
    const arma::Mat<Ele1>& M1,
    const arma::SpMat<Ele2>& M2,
    const unsigned col1,
    const unsigned col2
);

template <typename Ele1, typename Ele2>
inline double innerProd(
    const arma::SpMat<Ele1>& M1,
    const arma::SpMat<Ele2>& M2,
    const unsigned col1,
    const unsigned col2
);

template <typename Ele, typename PARAMTYPE, typename XtrVType>
inline void maintainW(
    PARAMTYPE& W,
    const XtrVType& XtrV,
    const arma::SpMat<Ele>& Xtr,
    const unsigned idx,
    const unsigned sampleCol,
    const double lambda1,
    const double lambda2,
    const double grad
);

template <typename DATATYPE, typename PARAMTYPE, typename EMBEDTYPE>
class pdenWithEmbedding: public EmbedBase <DATATYPE, EMBEDTYPE>{
    public:
    pdenWithEmbedding():
        EmbedBase<DATATYPE, EMBEDTYPE>(),
        lambda1(0.1),
        lambda2(0.5)
    {}

    pdenWithEmbedding(double lambda1_, double lambda2_):
        EmbedBase<DATATYPE, EMBEDTYPE>(),
        lambda1(lambda1_),
        lambda2(lambda2_)
    {}

    void train(
        const DATATYPE& X,
        const arma::icolvec& y,
        const unsigned nIter,
        const unsigned nPostIter,
        const unsigned embedDim,
        const std::string embedType,
        const arma::icolvec& sortedFlocs
    );

    double computeRegressionError(
        const DATATYPE& X, 
        const arma::icolvec& y
    );
    
    void predict(const DATATYPE& X, arma::icolvec& preds);

    void solve(
        const DATATYPE& Xtr,
        const arma::uvec& yIdx,
        arma::fmat& XtrV,
        arma::frowvec& Q,
        const int nIter,
        std::vector<std::set<unsigned>>& activeSets
    );

    void postSolve(
        const DATATYPE& Xtr,
        const arma::uvec& yIdx,
        const int nIter,
        const std::vector<std::set<unsigned>>& activeSets
    );
    private:
    double lambda1; //parameter for l1 regularization
    double lambda2; //parameter for l2 regularization
    PARAMTYPE W;
    EMBEDTYPE Vtr;  // transpose of dual variable
    std::vector<std::unordered_map<unsigned, unsigned>> feat2PostLoc;
    std::vector<PARAMTYPE> weightsVec;
    arma::frowvec featNorm;
};

template <typename ET, typename Ele1, typename Ele2>
inline unsigned selectIndex(
    const arma::SpMat<Ele1>& Xtr,
    const arma::Mat<Ele2>& W,
    const ET& embedM,
    const unsigned labelCol,
    const unsigned sampleIdx,
    ET& predApprox
) {
    auto it = Xtr.begin_col(sampleIdx);
    auto end = Xtr.end_col(sampleIdx);
    for (; it != end; ++it) predApprox += (*it) * W.row(it.row());
    return (arma::abs(embedM.col(labelCol).t() - predApprox)).index_max();
}

template <typename ET, typename Ele1, typename Ele2>
inline unsigned selectIndex(
    const arma::SpMat<Ele1>& Xtr,
    const arma::SpMat<Ele2>& W,
    const ET& embedM,
    const unsigned labelCol,
    const unsigned sampleIdx
) {
    ET diffAbs(embedM.col(labelCol).t());
    unsigned nEmbed = embedM.n_rows;
    for (unsigned k = 0; k < nEmbed; ++k) {
        auto it = Xtr.begin_col(sampleIdx);
        auto end = Xtr.end_col(sampleIdx);
        auto WIt = W.begin_col(k);
        auto WEnd = W.end_col(k);
        for (; (it!=end) && (WIt!=WEnd);) {
            if (it.row() < WIt.row()) {++it; continue;}
            if (it.row() > WIt.row()) {++WIt; continue;}
            diffAbs(k) -= (*it) * (*WIt);
            ++it; ++WIt;
        }
    }
    diffAbs = arma::abs(diffAbs);
    return diffAbs.index_max();
}

/* select active variable by importance sampling*/
template <typename ET, typename Ele>
inline unsigned selectIndex(
    const std::vector<float>& sampleAbs,
    const std::vector<int>& samplePos,
    const std::vector<unsigned>& rowIdx,
    const float l1norm,
    const unsigned featSampleNum,
    const arma::Mat<Ele>& W,
    const ET& embedM,
    const unsigned labelCol,
    ET& predApprox,
    std::default_random_engine& generator
) {
    std::discrete_distribution<unsigned> distr(
        sampleAbs.begin(),
        sampleAbs.end()
    );
    for (unsigned i = 0; i < featSampleNum; i++) {
        unsigned num = distr(generator);
        if (samplePos[num] > 0) predApprox += W.row(rowIdx[num]);
        else predApprox -= W.row(rowIdx[num]);
    }
    predApprox *= (l1norm / featSampleNum);
    ET diffAbs(arma::abs(embedM.col(labelCol).t() - predApprox));
    return diffAbs.index_max();
}

template <typename Ele1, typename Ele2>
inline double innerProd(
    const arma::Mat<Ele1>& M1,
    const arma::SpMat<Ele2>& M2,
    const unsigned col1,
    const unsigned col2
) {
    double res = 0.0;
    auto it = M2.begin_col(col2);
    auto end = M2.end_col(col2);
    for (; it != end; ++it) res += (*it) * M1(it.row(), col1);
    return res;
}

template <typename Ele1, typename Ele2>
inline double innerProd(
    const arma::SpMat<Ele1>& M1,
    const arma::SpMat<Ele2>& M2,
    const unsigned col1,
    const unsigned col2
) {
    double res = 0.0;
    auto it1 = M1.begin_col(col1);
    auto end1 = M1.end_col(col1);
    auto it2 = M2.begin_col(col2);
    auto end2 = M2.end_col(col2);
    for (; (it1!=end1)&&(it2!=end2);) {
        if (it1.row() > it2.row()) {++it2; continue;}
        if (it1.row() < it2.row()) {++it1; continue;}
        res += (*it1) * (*it2); ++it1; ++it2;
    }
    return res;
}

template <typename Ele, typename PT, typename XtrVType>
inline void maintainW(
    PT& W,
    XtrVType& XtrV,
    const arma::SpMat<Ele>& Xtr,
    const unsigned idx,
    const unsigned sampleCol,
    const double lambda1,
    const double lambda2,
    const double grad
) {
    double twoLambda2 = 2 * lambda2;
    double minusLambda1 = -lambda1;
    auto it = Xtr.begin_col(sampleCol);
    auto end = Xtr.end_col(sampleCol);
    for (; it != end; ++it) {
        XtrV(it.row(), idx) -= grad * (*it);
        double val = XtrV(it.row(), idx);
        if (val > lambda1) W(it.row(), idx) = (val - lambda1) / twoLambda2;
        else if (val < minusLambda1) 
            W(it.row(), idx) = (val + lambda1) / twoLambda2;
        else W(it.row(), idx) = 0;
    }
}

template <typename DT, typename PT, typename ET>
inline void pdenWithEmbedding<DT, PT, ET>::train(
    const DT& X,
    const arma::icolvec& y,
    const unsigned nIter,
    const unsigned nPostIter,
    const unsigned embedDim,
    const std::string embedType,
    const arma::icolvec& sortedFlocs_
) {
    this->check_input(y, sortedFlocs_);
    int nClasses = this->sorted_labels.n_rows;
    GetEmbeddingMatrix(embedType, nClasses, embedDim, this->embed_m);
    arma::uvec yIdx;
    this->y2yIdx(y, yIdx);

    int nData = X.n_rows;
    int nFeat = X.n_cols;
    int nEmbed = embedDim;
    DT Xcopy(X);
    //this->featNorm = arma::max(arma::abs(Xcopy), 0);
    //for (unsigned j = 0; j < nFeat; ++j) Xcopy.col(j) /= this->featNorm(j);
    DT Xtr = Xcopy.t().eval();
    
    // precalculation/allocation
    this->Vtr.zeros(nEmbed, nData);
    this->W.zeros(nFeat, nEmbed);

    // TODO: allow an option to specify sparse/dense XtrV
    arma::fmat XtrV(nFeat, nEmbed, arma::fill::zeros);
    arma::frowvec Q = arma::sum(arma::square(Xtr), 0) / this->lambda2 + 1;
    std::vector<std::set<unsigned>> activeSets(nData);
    this->solve(Xtr, yIdx, XtrV, Q, nIter, activeSets); 
    this->postSolve(Xtr, yIdx, nPostIter, activeSets);
}

template <typename DT, typename PT, typename ET>
inline double pdenWithEmbedding<DT, PT, ET>::computeRegressionError(
        const DT& X, 
        const arma::icolvec& y
){
    ET yEmbed;
    this->ConstructEmbeddedTarget(y, yEmbed); 
    unsigned nFeat = this->featNorm.n_cols;
    DT Xcopy(X);
    //for (unsigned j = 0; j < nFeat; ++j) Xcopy.col(j) /= this->featNorm(j);
    ET regPreds(Xcopy * this->W);
    return arma::accu(arma::pow(regPreds - yEmbed, 2)) / regPreds.n_rows;
}

template <typename DT, typename PT, typename ET>
inline void pdenWithEmbedding<DT, PT, ET>::predict(
    const DT& X,  //use std::move if X no longer needed
    arma::icolvec& preds
){
    unsigned nFeat = this->featNorm.n_cols;
    DT Xcopy(X);
    //for (unsigned j = 0; j < nFeat; ++j) Xcopy.col(j) /= this->featNorm(j);
    ET regPreds(Xcopy * this->W);
    this->PredictHelper(regPreds, preds); 
}

template <typename DT, typename PT, typename ET>
inline void pdenWithEmbedding<DT, PT, ET>::solve(
    const DT& Xtr,
    const arma::uvec& yIdx,
    arma::fmat& XtrV,
    arma::frowvec& Q,
    const int nIter,
    std::vector<std::set<unsigned>>& activeSets
) {
    std::cout << "nonzeros in X: " << countNNZ(Xtr) << " out of ";
    std::cout << Xtr.n_rows * Xtr.n_cols << std::endl;

    const unsigned nData = Xtr.n_cols;
    /* initialize for importance sampling*/
    std::vector<float> normSample(nData, 0.0);
    std::vector<std::vector<float>> XtrAbs(nData);
    std::vector<std::vector<int>> XtrPos(nData);
    std::vector<std::vector<unsigned>> XtrRowIdx(nData);
    auto it = Xtr.begin();
    auto endIt = Xtr.end();
    for (; it != endIt; ++it) {
        float absVal = fabs(*it);
        normSample[it.col()] += absVal;
        XtrAbs[it.col()].push_back(absVal);
        XtrPos[it.col()].push_back(((*it) > 0) * 2 - 1);
        XtrRowIdx[it.col()].push_back(it.row());
    }
    std::default_random_engine rgn;
    /* initialize for importance sampling*/

    #ifdef MEASURE_TIME
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    #endif
    #ifdef MEASURE_TIME
    double selectionTime = 0.0;
    double gradTime = 0.0;
    double maintainWTime = 0.0;
    #endif
    
    for (unsigned c = 0; c < nIter; c++) {
        std::cout << "Iteration " << c << std::endl;
        arma::uvec samples = arma::randperm(nData);
        for (unsigned j = 0; j < nData; j++) {
            unsigned i = samples(j);
            DEBUG_MSG("data point " << i << " is sampled");
            DEBUG_MSG("updating active set");

            #ifdef MEASURE_TIME
            start = std::chrono::steady_clock::now();
            #endif
            /*select a dual variable*/
            ET prodCache(1, this->embed_m.n_rows, arma::fill::zeros);
            /*unsigned idx = selectIndex(
                Xtr,
                this->W,
                this->embed_m,
                yIdx(i),
                i,
                prodCache
            );*/
            // TODO: calculate the speed up rate
            unsigned idx = selectIndex(
                XtrAbs[i],
                XtrPos[i],
                XtrRowIdx[i],
                normSample[i],
                std::max((int)(XtrAbs[i].size() / 5), 5),
                this->W,
                this->embed_m,
                yIdx(i),
                prodCache,
                rgn 
            );
            /*select a dual variable done*/
            #ifdef MEASURE_TIME
            end = std::chrono::steady_clock::now();
            selectionTime += std::chrono::duration_cast
                <std::chrono::microseconds> (end-start).count() / 1e6;
            #endif

            activeSets[i].emplace(idx);
            std::vector<std::set<unsigned>::iterator> toRemove;
            std::vector<double> deltaVi;
            deltaVi.reserve(activeSets.size());
            DEBUG_MSG("active set updated by adding index " << idx);
            DEBUG_MSG("updating active dual variables");

            #ifdef MEASURE_TIME
            start = std::chrono::steady_clock::now();
            #endif
            for (auto iter = activeSets[i].begin();
                iter != activeSets[i].end();
                iter++) 
            {
                /*update dual variables*/
                unsigned activeIdx = *iter;
                double currGrad = 0.5 * Vtr(activeIdx, i)
                    - this->embed_m(activeIdx, yIdx(i))
                    + prodCache(activeIdx);
                    //+ innerProd(this->W, Xtr, activeIdx, i);
                currGrad /= (Q(i) / 2.0);
                Vtr(activeIdx, i) -= currGrad;
                /*update dual variables done*/
                /*maintain active set*/
                if (fabs(Vtr(activeIdx, i)) < ZERO_TOL) {
                    double val = Vtr(activeIdx, i);
                    Vtr(activeIdx, i) = 0;
                    toRemove.push_back(std::move(iter));
                    deltaVi.push_back(val+currGrad);
                } 
                else deltaVi.push_back(currGrad);
                /*maintain active set done*/
            }
            #ifdef MEASURE_TIME
            end = std::chrono::steady_clock::now();
            gradTime += std::chrono::duration_cast
                <std::chrono::microseconds> (end-start).count() / 1e6;
            #endif
            DEBUG_MSG("active dual variables updated");
            DEBUG_MSG("maintaining primal variables");

            #ifdef MEASURE_TIME
            start = std::chrono::steady_clock::now();
            #endif
            /*maintain W*/
            unsigned activePos = 0;
            for (auto iter = activeSets[i].begin();
                iter != activeSets[i].end();
                iter++)
            {
                double delta = deltaVi[activePos++];
                if (fabs(delta) < ZERO_TOL) continue;
                maintainW(
                    this->W,
                    XtrV,
                    Xtr,
                    *iter,
                    i,
                    this->lambda1,
                    this->lambda2,
                    delta
                );
            }
            /*maintain W done*/
            /*maintain the active set*/
            for (auto& iter: toRemove) {activeSets[i].erase(iter);}
            /*maintain the active set done*/ 
            #ifdef MEASURE_TIME
            end = std::chrono::steady_clock::now();
            maintainWTime += std::chrono::duration_cast
                <std::chrono::microseconds> (end-start).count() / 1e6;
            #endif
            DEBUG_MSG("primal variables updated");
        }
    } 
    #ifdef MEASURE_TIME
    std::cout << "selection time: " << selectionTime << "s" << std::endl;
    std::cout << "gradients update time: " << gradTime << "s" << std::endl;
    std::cout << "maintain W time: " << maintainWTime << "s" << std::endl;
    #endif
    Vtr.clean(ZERO_TOL);
    this->W.clean(ZERO_TOL);
    XtrV.clean(ZERO_TOL);
    std::cout << "nonzeros in V: " << countNNZ(Vtr) << " out of ";
    std::cout << Vtr.n_rows * Vtr.n_cols << std::endl;
    std::cout << "nonzeros in W: " << countNNZ(this->W) << " out of ";
    std::cout << this->W.n_rows * this->W.n_cols << std::endl;
    std::cout << "nonzeros in Xtr*V: " << countNNZ(XtrV) << " out of ";
    std::cout << XtrV.n_rows * XtrV.n_cols << std::endl;
    std::cout << "erro in XtrV: ";
    std::cout << arma::norm(Xtr*Vtr.t() - XtrV, "fro") << std::endl;
}

template <typename DT, typename PT, typename ET>
inline void pdenWithEmbedding<DT, PT, ET>::postSolve(
    const DT& Xtr,
    const arma::uvec& yIdx,
    const int nIter,
    const std::vector<std::set<unsigned>>& activeSets
) {
    const unsigned nData = Xtr.n_cols;
    const unsigned nEmbed = this->embed_m.n_rows;
    std::vector<std::vector<std::vector<unsigned>>> Xktrs(
        nEmbed,
        std::vector<std::vector<unsigned>>(nData)
    );
    std::vector<std::vector<std::vector<double>>> XktrVal(
        nEmbed,
        std::vector<std::vector<double>>(nData)
    );
    std::vector<std::vector<double>> Q(nEmbed, std::vector<double>(nData));
    for (unsigned k = 0; k < nEmbed; ++k) {
        for (unsigned i = 0; i < nData; ++i) {
            auto it = Xtr.begin_col(i);
            auto end = Xtr.end_col(i);
            double qki = 0.0;
            for (; it != end; ++it) {
                if (fabs(this->Vtr(k, i)) < ZERO_TOL) break; 
                if (fabs(this->W(it.row(), k)) < ZERO_TOL) continue;
                Xktrs[k][i].push_back(it.row());
                XktrVal[k][i].push_back(*it);
                qki += (*it) * (*it);
            }
            Q[k][i] = qki / this->lambda2 + 1;
        }
    }

    PT W0(this->W);
    PT Wcopy(this->W);
    
    double twoLambda2 = 2 * lambda2;
    for (unsigned c = 0; c < nIter; c++) {
        std::cout << "Post Solve Iteration " << c << std::endl;
        // TODO: randomly permute samples
        for (unsigned i = 0; i < nData; i++) {
            for (auto iter = activeSets[i].begin();
                iter != activeSets[i].end();
                iter++) 
            {
                /*update dual variables*/
                unsigned activeIdx = *iter;
                double inner = 0.0;
                auto it = Xktrs[activeIdx][i].begin();
                auto end = Xktrs[activeIdx][i].end();
                auto valIt = XktrVal[activeIdx][i].begin();
                for (; it != end; ++it, ++valIt) {
                    inner += (*valIt) * Wcopy(*it, activeIdx);
                }
                double currGrad = 0.5 * Vtr(activeIdx, i)
                    - this->embed_m(activeIdx, yIdx(i))
                    + inner;
                currGrad /= Q[activeIdx][i];
                Vtr(activeIdx, i) -= currGrad;
                /*update dual variables done*/

                /*maintain W*/
                it = Xktrs[activeIdx][i].begin();
                end = Xktrs[activeIdx][i].end();
                valIt = XktrVal[activeIdx][i].begin();
                for (; it != end; ++it, ++valIt) {
                    this->W(*it, activeIdx) -= currGrad*(*valIt)/twoLambda2; 
                }
                /*maintain W done*/
            }
            for (auto iter = activeSets[i].begin();
                iter != activeSets[i].end();
                iter++) {
                auto it = Xktrs[*iter][i].begin();
                auto end = Xktrs[*iter][i].end();
                for (; it != end; ++it) Wcopy(*it, *iter) = this->W(*it, *iter);
            }
        }
    }
    Vtr.clean(ZERO_TOL);
    this->W.clean(ZERO_TOL);
    std::cout << "nonzeros in V: " << countNNZ(Vtr) << " out of ";
    std::cout << Vtr.n_rows * Vtr.n_cols << std::endl;
    std::cout << "nonzeros in W: " << countNNZ(this->W) << " out of ";
    std::cout << this->W.n_rows * this->W.n_cols << std::endl;
}
