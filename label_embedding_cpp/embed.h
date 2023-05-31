#ifdef DEBUG
#define DEBUG_MSG(str) do {std::cerr << "DEBUG: " << str << std::endl;} \
    while (false)
#else
#define DEBUG_MSG(str) do { } while (false)
#endif

#include <algorithm>
#include <armadillo>
#include <iostream>
#include <limits>
#include <string>

template <class Ele>
inline void GetEmbeddingMatrix(const std::string embed_type,
                               const int num_classes,  // n_cols
                               const int embed_dim,   // n_rows
                               arma::Mat<Ele>& EmbedM){
    if (embed_type == "Gaussian")  // Gaussian sketch 
    EmbedM = arma::randn<arma::Mat<Ele>>(embed_dim, num_classes + 1)
        / sqrt(embed_dim);
    else if (embed_type == "Rademacher"){ // Rademacher Sketch:
        EmbedM = arma::randi<arma::Mat<Ele>>(embed_dim,
                                             num_classes + 1,
                                             arma::distr_param(0, 1));
        EmbedM = (EmbedM * 2 - 1) / sqrt(embed_dim);
    }
    else throw std::invalid_argument("Unkwown JLT Matrix Type");

}

template <class Ele>
inline void find1NN(const arma::Mat<Ele>& reg_out,
                    const arma::Mat<Ele>& embed_m,
                    bool good_label,
                    const arma::icolvec& sorted_labels,
                    arma::ucolvec& preds){
    // preds here are actually indices of predicted labels
    int n_data = reg_out.n_rows;
    int out_dim = reg_out.n_cols;
    int num_classes = embed_m.n_cols;
    // recover label by matrix multiplication
    arma::Col<Ele> out_norm_sq = arma::sum(arma::pow(reg_out, 2), 1);
    arma::Row<Ele> embed_norm_sq = arma::sum(arma::pow(embed_m, 2), 0);
    
    arma::Mat<Ele> out_expanded(n_data, out_dim+2);
    out_expanded.submat(0, 0, n_data-1, out_dim-1) = reg_out;
    out_expanded.col(out_dim).ones();
    out_expanded.col(out_dim+1) = out_norm_sq;
    
    arma::Mat<Ele> embed_expanded(out_dim+2, num_classes);
    embed_expanded.submat(0, 0, out_dim-1, num_classes-1) = -2 * embed_m;
    embed_expanded.row(out_dim) = embed_norm_sq;
    embed_expanded.row(out_dim+1).ones();

    // directly do matrix multiplication may comsume too much memory
    //preds = arma::index_min(out_expanded * embed_expanded, 1);

    preds.zeros(n_data);
    arma::Mat<Ele> out_expanded_t = out_expanded.t().eval();
    Ele* out_t_ptr = out_expanded_t.memptr();
    Ele* embed_ptr = embed_expanded.memptr();
    auto* preds_ptr = preds.memptr();
    #pragma omp parallel for
    for (unsigned i = 0; i < n_data; ++i){
        Ele min_dist = std::numeric_limits<Ele>::max();
        for (unsigned j = 0; j < num_classes; ++j) {
            Ele dist = 0;
            for (unsigned k = 0; k < out_dim + 2; ++k) {
                Ele temp1 = *(out_t_ptr+i*(out_dim+2)+k);
                Ele temp2 = *(embed_ptr+j*(out_dim+2)+k);
                dist += temp1 * temp2; 
            }
            if (dist < min_dist) {
                min_dist = dist;
                *(preds_ptr + i) = j;
            }
        }
    }
}

template <typename DATATYPE, typename EMBEDTYPE>
class EmbedBase{
    public:
    EmbedBase(): good_label(false), good_feature(false) {}
     
    const arma::icolvec& get_flocs() {return sorted_flocs;}
    bool get_feat_status() {return good_feature;}

    template <typename Ele>
    void syncData(const arma::Mat<Ele>& X) {return;}

    template <typename Ele>
    void syncData(const arma::SpMat<Ele>& X) {X.sync();}
    void PredictHelper(const EMBEDTYPE& reg_out, arma::icolvec& preds);    

    protected:
    void check_input(
        const arma::icolvec& y,
        const arma::icolvec& sorted_flocs_ 
    );

    void y2yIdx(const arma::icolvec& y, arma::uvec& yIdx);
    void ConstructEmbeddedTarget(const arma::icolvec& y, EMBEDTYPE& Y_Embed);

    EMBEDTYPE embed_m;
    arma::icolvec sorted_labels;
    arma::icolvec sorted_flocs;
    bool good_label; // indicating if label ranges from 0 to num_labels
    bool good_feature;
};

template <typename DATATYPE, typename EMBEDTYPE>
class EmbedMultiBase{
    public:
    EmbedMultiBase(unsigned K): K(K) {}
     
    template <typename Ele>
    void syncData(const arma::Mat<Ele>& X) {return;}

    template <typename Ele>
    void syncData(const arma::SpMat<Ele>& X) {X.sync();}

    EMBEDTYPE embed_m;
    unsigned K; // max size of eta(x)'s support

    protected:
    void ConstructEmbeddedTarget(
        const std::vector<std::vector<unsigned>>& y,
        EMBEDTYPE& Y_Embed
    );

};

template <typename DT, typename ET>
inline void EmbedBase<DT, ET>::PredictHelper(
    const ET& reg_out,
    arma::icolvec& preds
){
    // find the 1NN
    arma::ucolvec pred_indices;
    find1NN(
        reg_out,
        this->embed_m,
        this->good_label,
        this->sorted_labels,
        pred_indices
    );
    preds = arma::icolvec(pred_indices.n_rows);
    if (this->good_label){
        preds = arma::conv_to<arma::icolvec>::from(std::move(pred_indices));
        return;
    }
    std::transform(
        pred_indices.begin(),
        pred_indices.end(),
        preds.begin(),
        [this](size_t idx) {return this->sorted_labels(idx);}
    );
}

template <typename DT, typename ET>
inline void EmbedBase<DT, ET>::check_input(
        const arma::icolvec& y,
        const arma::icolvec& sorted_flocs_ 
){
        // Check input
        // X should be of the shape (n_data, n_features)
        this->good_feature = (sorted_flocs_.n_elem == 0);
        this->sorted_flocs = std::move(sorted_flocs_);
        this->sorted_labels = arma::unique(y);
        if (this->sorted_labels.n_rows == (arma::max(this->sorted_labels)+1))
            this->good_label = true;
}

template <typename DT, typename ET>
inline void EmbedBase<DT, ET>::y2yIdx(const arma::icolvec& y, arma::uvec& yIdx){
    if (this->good_label) {
        yIdx = arma::conv_to<arma::uvec>::from(y);
        return;
    }
    yIdx = arma::uvec(y.n_rows);
    for (int i = 0; i < y.n_rows; i++){
        auto label_iter = std::lower_bound(this->sorted_labels.begin(), 
                                           this->sorted_labels.end(),
                                           y(i));
        yIdx(i) = label_iter - this->sorted_labels.begin();
    }
}

template <typename DT, typename ET>
inline void EmbedBase<DT, ET>::ConstructEmbeddedTarget(
    const arma::icolvec& y,
    ET& Y_Embed
){
    Y_Embed = ET(y.n_rows, this->embed_m.n_rows);
    //if (this->good_label) {
        for (int i = 0; i < y.n_rows; i++) { 
            Y_Embed.row(i) = this->embed_m.col(y(i)).t();
        }
        return;
    //}
    /*
    for (int i = 0; i < y.n_rows; i++){
        auto label_iter = std::lower_bound(this->sorted_labels.begin(), 
                                           this->sorted_labels.end(),
                                           y(i));
        Y_Embed.row(i) = embed_m.col(label_iter-this->sorted_labels.begin()
            ).t();
    }*/
}

template <typename DT, typename ET>
inline void EmbedMultiBase<DT, ET>::ConstructEmbeddedTarget(
    const std::vector<std::vector<unsigned>>& y,
    ET& yEmbed
){
    yEmbed = ET(this->embed_m.n_rows, y.size(), arma::fill::zeros);
    const unsigned nData = y.size();
    for (unsigned i = 0; i < nData; ++i) {
        for (unsigned label: y[i])
            if (label < this->embed_m.n_cols)
                yEmbed.col(i) += this->embed_m.col(label);
    }
    yEmbed = yEmbed.t().eval();
}

template <typename Ele, typename Idx>
inline void omp(
    Ele *  Aptr,  // measurement matrix, column-major order
    Ele *  xptr,  // observation
    Idx * const cols,  // columns idxs of yptr, size=max_iter 
    Ele * const yptr,  // recovered signal, size=max_iter
    const unsigned n_rows,  // num of rows of A = num of observations
    const unsigned n_cols,  // num of cols of A = num of variables
    const unsigned max_iters  // y will have this number of signals
) {
    arma::Mat<Ele> A(Aptr, n_rows, n_cols, false, true);
    arma::Row<Ele> r(xptr, n_rows);
    arma::Col<Ele> x(xptr, n_rows, false, true);
    arma::Row<Ele> ANormSq = arma::sum(arma::square(A), 0);
    arma::Mat<Ele> corrMat(max_iters, max_iters);
    arma::Col<Ele> lsqTarget(max_iters);
    for (unsigned k=0; k < max_iters; ++k) {
        arma::Row<Ele> corr = (r * A) / ANormSq;
        unsigned newCol = corr.index_max();
        cols[k] = newCol;
        lsqTarget[k] = arma::sum(x % A.unsafe_col(newCol));
        for (unsigned j=0; j<=k; ++j) {
            unsigned col = cols[j];
            corrMat(j, k) = arma::sum(A.unsafe_col(col) % A.unsafe_col(newCol));
            corrMat(k, j) = corrMat(j, k);
        }
        arma::Col<Ele> y = arma::solve(
                            corrMat.submat(0, 0, k, k), 
                            lsqTarget.head(k+1)
                            );
        r = (x - A.cols(arma::Col<Idx>(cols, k+1, false, true)) * y).t();
        if (k == max_iters - 1)
            memcpy(yptr, y.memptr(), sizeof(Ele) * max_iters);
    }
}
