#include <armadillo>
#include "embed.h"
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <string>

template <typename DATATYPE, typename PARAMTYPE, typename EMBEDTYPE>
class ENwithEmbedding: public EmbedBase<DATATYPE, EMBEDTYPE> {
    public:
    ENwithEmbedding(double lambda, double alpha):
        EmbedBase<DATATYPE, EMBEDTYPE>(),
        C1(lambda * alpha),
        C2((1-alpha) * lambda * 0.5)
    {};
    
    void train(
        DATATYPE X,
        const arma::icolvec& y,
        const int embed_dim,
        const std::string embed_type,
        const int num_iters,
        const arma::icolvec& sorted_flocs_,
        const double tol=1e-6
    );  // train the model by cd

    void predict(
        DATATYPE X,
        arma::icolvec& preds
    );  // make predictions
    
    double computeOBJ(
        const DATATYPE& X,
        const arma::icolvec& y
    );  // compute optimization objective
    
    private:
    double C1; // l1 regularization parameter
    double C2; // l2 regularization parameter
    PARAMTYPE W;
    DATATYPE mean;
    DATATYPE std;
};

template <typename DT, typename PT, typename ET>  // data/param/embedding types
inline void ENwithEmbedding<DT, PT, ET>::train(
    DT X, // use std::move if X is no longer needed
    const arma::icolvec& y, 
    const int embed_dim,
    const std::string embed_type,
    const int num_iters,
    const arma::icolvec& sorted_flocs_,
    const double tol
) {
    ET Y_Embed;
    this->check_input(y, sorted_flocs_);
    int num_classes = this->sorted_labels.n_rows;
    GetEmbeddingMatrix(embed_type, num_classes, embed_dim, this->embed_m);
    this->ConstructEmbeddedTarget(y, Y_Embed);

    //normalize data
    mean = arma::mean(X, 0);
    std = arma::stddev(X, 0, 0);
    X.each_row() -= mean;
    X.each_row() /= std;
     
    DEBUG_MSG("good label = " << this->good_label);    

    int num_data = X.n_rows;
    int num_features = X.n_cols; 
    // initialize weights
    // TODO: implement initialization for sparse matrix
    W.zeros(num_features, embed_dim);
    // precalculate the residual and squared column nomrs of X
    // TODO: add bias later
    // TODO: may need to refactor the template parameters
    ET R = Y_Embed - X * W;
    DT sq_sum_cols = arma::mean(arma::square(X), 0);
    // initialize variables
    double loss = arma::norm(R, "fro")/ (2*num_data);
    double l2_r = 0.5 * C2 * arma::norm(W, "fro");
    double l1_r = C1 * arma::accu(arma::abs(W));
    double obj = loss + l2_r + l1_r;
    // now let's do coordinate descent:
    std::cout << "CD main loop begins" << std::endl;
    for (int outer=0;outer<num_iters;outer++){
        for (int i=0;i<num_features;i++){
            ET temp = X.col(i).t() * R + sq_sum_cols(i) * W.row(i);
            temp /= num_data;
            temp.clean(C1);
            PT old_wi = W.row(i);
            W.row(i) = arma::sign(temp) % (arma::abs(temp) - C1)
                                / (C2 + sq_sum_cols(i));
            // update R for the next iteration
            R += X.col(i) * (old_wi - W.row(i));
        }
        double obj_pre = obj;
        loss = arma::norm(R, "fro")/ (2*num_data);
        l2_r = 0.5 * C2 * arma::norm(W, "fro");
        l1_r = C1 * arma::accu(arma::abs(W));
        obj = loss + l2_r + l1_r;
        std::cout << outer << "th iteration, ";
        std::cout << "objective: " << obj << std::endl;
        
        DEBUG_MSG("loss: " << loss);
        DEBUG_MSG("l1 regularization = " << l1_r);
        DEBUG_MSG("l2 regularization = " << l2_r);
        if ((outer > 0) && (fabs(obj - obj_pre) < tol)) {
            std::cout << "Model converges! Training completes." << std::endl;
            break; 
        }
        if (outer == num_iters-1)
            std::cout << "Model fails to converge." << std::endl;
    }  
}

template <typename DT, typename PT, typename ET>  // data/param/embedding types
inline void ENwithEmbedding<DT, PT, ET>::predict(
    DT X,  //use std::move if X no longer needed
    arma::icolvec& preds
) {
    X.each_row() -= mean;
    X.each_row() /= std;
    ET reg_out = X * W;
    this->PredictHelper(reg_out, preds); 
}
