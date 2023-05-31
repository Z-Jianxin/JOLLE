#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

class InputParser{
    public:

        InputParser (int argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }

        const std::string getCmdOption(const std::string option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }

        bool cmdOptionExists(const std::string option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }

    private:

        std::vector <std::string> tokens;
};

template <typename DATATYPE>
void load_train(DATATYPE& X,
                arma::icolvec& y,
                std::string& data_path,
                arma::icolvec& sorted_flocs){
    std::string next;
    std::ifstream data_f(data_path);
    std::unordered_set<int> flocs_set;
    int num_data = 0;
    int num_feat = 0;
    int max_floc = 0;
    while (data_f >> next) {
        size_t loc = next.find(":");
        if ( loc == std::string::npos) num_data++;
        else {
            int col_idx = std::stoi(next.substr(0, loc));
            max_floc = std::max(col_idx, max_floc);
            flocs_set.emplace(col_idx);
        }
    }
    num_feat = flocs_set.size();
    bool good_feat = (num_feat == max_floc+1);
    if (!good_feat) {
        sorted_flocs = arma::icolvec(num_feat);
        auto iter = flocs_set.begin();
        for (int i=0;i<num_feat;i++){sorted_flocs(i)=*iter;iter++;}
        sorted_flocs = arma::sort(sorted_flocs);
    }
    X.zeros(num_data, num_feat);
    y = arma::icolvec(num_data);

    data_f.clear();  // clear end of file state
    data_f.seekg(0);  // reset cursor position 
    int row_idx = -1;
    while (data_f >> next) {
        size_t loc = next.find(':');
        if ( loc == std::string::npos) y(++row_idx) = std::stoi(next);
        else {
            int col_idx = std::stoi(next.substr(0, loc));
            if (!good_feat){
                auto iter = std::lower_bound(sorted_flocs.begin(),
                                             sorted_flocs.end(),
                                             col_idx);
                col_idx = iter - sorted_flocs.begin();
            }
            X(row_idx, col_idx) = std::stod(next.substr(loc+1));
        }
    }
    data_f.close();
}

template <typename DATATYPE>
void load_test(DATATYPE& X,
               arma::icolvec& y,
               std::string& data_path,
               const arma::icolvec& sorted_flocs,
               bool good_feat) {
    std::string next;
    std::ifstream data_f(data_path);
    int num_data = 0;
    int num_feat = sorted_flocs.n_rows;
    while (data_f >> next) {
        size_t loc = next.find(":");
        if ( loc == std::string::npos) num_data++;
        if (good_feat) {
            int col_idx = std::stoi(next.substr(0, loc));
            num_feat = std::max(num_feat, col_idx+1);
        }
    }
    X.zeros(num_data, num_feat);
    y = arma::icolvec(num_data);
    data_f.clear();  // clear end of file state
    data_f.seekg(0);  // reset cursor position 
    int row_idx = -1;
    while (data_f >> next) {
        size_t loc = next.find(':');
        if ( loc == std::string::npos) y(++row_idx) = std::stoi(next);
        else {
            int col_idx = std::stoi(next.substr(0, loc));
            if (!good_feat){
                auto iter = std::lower_bound(sorted_flocs.begin(),
                                             sorted_flocs.end(),
                                             col_idx);
                col_idx = iter - sorted_flocs.begin();
            }
            X(row_idx, col_idx) = std::stod(next.substr(loc+1));
        }
    }
    data_f.close();
}

template <typename DATATYPE>
void insertOne(DATATYPE& X) {
    DATATYPE X1(X.n_rows, X.n_cols + 1);
    X1.cols(0, X.n_cols - 1) = X;
    for (unsigned j = 0; j < X.n_rows; ++j) X1(j, X.n_cols) = 1;
    X = std::move(X1);
}

template<typename EleType>
unsigned loadSparse(
    arma::SpMat<EleType>& X,
    std::vector<std::vector<unsigned>>& labelNames,
    std::string& dataPath
) { 
    // the function returns the number of unique labels
    std::string line;
    std::ifstream dataFile(dataPath);
    std::vector<unsigned> rowLocs;
    std::vector<unsigned> colLocs;
    std::vector<EleType> valVec;
    const char delim = ':';
    const char labelDelim = ',';
    unsigned currRow = 0;
    unsigned maxLabel = 0;
    while(std::getline(dataFile, line)) {
        std::istringstream linestream(line);
        std::vector<unsigned> labels;
        std::string next;

        linestream >> next;
        std::istringstream labelstream(next);
        std::string labelStr;
        while(std::getline(labelstream, labelStr, labelDelim)) {
            unsigned label = std::stoul(labelStr);
            labels.push_back(label);
            maxLabel = std::max(maxLabel, label);
        }
        while (linestream >> next) {
            size_t delimLoc = next.find(delim);
            unsigned colLoc = std::stoi(next.substr(0, delimLoc));
            EleType val = std::stod(next.substr(delimLoc+1));
            rowLocs.push_back(currRow);
            colLocs.push_back(colLoc);
            valVec.push_back(val);
        }
        labelNames.push_back(std::move(labels));
        currRow++;
    }
    unsigned nnz = valVec.size();
    arma::umat locations(2, nnz);
    locations.row(0) = arma::conv_to<arma::urowvec>::from(rowLocs);
    locations.row(1) = arma::conv_to<arma::urowvec>::from(colLocs);
    X = arma::SpMat<EleType>(
        locations,
        arma::conv_to<arma::fvec>::from(valVec)
    );
    return maxLabel + 1;
}

double hammingDist(
    std::vector<std::vector<unsigned>>& preds,
    std::vector<std::vector<unsigned>>& labels
) {
    double dist=0; 
    #pragma omp parallel for reduction(+: dist) 
    for (unsigned i=0; i<preds.size(); ++i) {
        std::sort(preds[i].begin(), preds[i].end());
        std::sort(labels[i].begin(), labels[i].end());
        std::vector<unsigned> fakePos(preds[i].size());
        std::vector<unsigned> fakeNeg(labels[i].size());
        auto itfp = std::set_difference(
            preds[i].begin(),
            preds[i].end(),
            labels[i].begin(),
            labels[i].end(),
            fakePos.begin()
        );
        auto itfn = std::set_difference(
                            labels[i].begin(),
                            labels[i].end(),
                            preds[i].begin(),
                            preds[i].end(),
                            fakeNeg.begin()
        );
        dist += (itfp - fakePos.begin()) + (itfn - fakeNeg.begin());
    }
    dist /= preds.size();
    return dist;
}

double precision(
    std::vector<std::vector<unsigned>>& preds,
    std::vector<std::vector<unsigned>>& labels
) {
    double prec=0; 
    #pragma omp parallel for reduction(+: prec) 
    for (unsigned i=0; i<preds.size(); ++i) {
        std::sort(preds[i].begin(), preds[i].end());
        std::sort(labels[i].begin(), labels[i].end());
        std::vector<unsigned> truePos(preds[i].size());
        auto it = std::set_intersection(
            preds[i].begin(),
            preds[i].end(),
            labels[i].begin(),
            labels[i].end(),
            truePos.begin()
        );
        prec += (double)(it - truePos.begin()) / preds[i].size();
    }
    prec /= preds.size();
    return prec;
}

void preds_from_probs(
    std::vector<std::vector<unsigned>>& preds,
    const unsigned long long * const labels,
    const float * const probs,
    const unsigned K, 
    const float thresh
) {
    const unsigned nData = preds.size();
    #pragma omp parallel for 
    for (unsigned i=0; i < nData; ++i){
        for (unsigned j=0; j < K; ++j) {
            if (probs[i*K + j] > thresh) preds[i].push_back((labels[i*K + j]));
        }
        if (preds[i].size() == 0) {
            preds[i].push_back(
                labels[std::max_element(probs+i*K, probs+(i+1)*K) - probs]
            );
        }
    }
}

void topk_from_probs(
    std::vector<std::vector<unsigned>>& preds,
    const unsigned long long * const labels,
    const float * const probs,
    const unsigned K,
    const unsigned k 
) {
    const unsigned nData = preds.size();
    #pragma omp parallel for 
    for (unsigned i=0; i < nData; ++i){
        std::vector<unsigned> indices(K);
        for (unsigned t=0; t < K; ++t) indices[t]=t;
        std::sort(
            indices.begin(),
            indices.end(),
            [probs, i, K](const unsigned a, const unsigned b) {
                return probs[i*K + a] > probs[i*K + b];
            }  // use > to get descending order
        );
        for (unsigned j=0; j<k; ++j) {
            preds[i].push_back(labels[i*K + indices[j]]);
        }    
    }
}
