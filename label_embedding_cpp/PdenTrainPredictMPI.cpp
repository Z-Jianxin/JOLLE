#include <mpi.h>
#include "pdenParallel.h"
#include <stdlib.h>
#include "util.h"

#ifdef SPARSE_DATA
#define DATATYPE arma::sp_fmat
#else
#define DATATYPE arma::fmat
#endif

#ifdef SPARSE_MODEL
#define PARAMTYPE arma::sp_fmat
#else
#define PARAMTYPE arma::fmat
#endif

struct Param{
    std::string trainDataPath;
    std::string testDataPath;
    int embedDim;
    std::string embedType;
    double lambda1;
    double lambda2;
    int nIter;
    int nPostIter;
};

void parsing(int argc, char** argv, Param & param){
    InputParser parser(argc, argv);
    if(parser.cmdOptionExists("-h")){
        std::cerr << "output some help information" << std::endl;
        exit(0);
    }

    const std::string lambda1 = parser.getCmdOption("-l1");
    if (!lambda1.empty()) param.lambda1 = std::stod(lambda1);
    else param.lambda1 = 1.0;

    const std::string lambda2 = parser.getCmdOption("-l2");
    if (!lambda2.empty()) param.lambda2 = std::stod(lambda2);
    else param.lambda2 = 0.5;
    
    const std::string nIter = parser.getCmdOption("-n");
    if (!nIter.empty()) param.nIter = std::stoi(nIter);
    else param.nIter = 20;

    const std::string nPostIter = parser.getCmdOption("-p");
    if (!nPostIter.empty()) param.nPostIter = std::stoi(nPostIter);
    else param.nPostIter = 10;

    const std::string trainDataPath = parser.getCmdOption("-trd");
    if (!trainDataPath.empty()) param.trainDataPath = trainDataPath;
    else{
        std::cerr << "need train data path" << std::endl;
        exit(1);
    }

    const std::string testDataPath = parser.getCmdOption("-ted");
    if (!testDataPath.empty()) param.testDataPath = testDataPath;
    else{
        std::cerr << "need test data path" << std::endl;
        exit(1);
    }
    
    const std::string embedDim = parser.getCmdOption("-e");
    if (!embedDim.empty()) param.embedDim = std::stoi(embedDim);
    else{
        std::cerr << "embedding dimension unspecified" << std::endl;
        exit(1);
    }

    const std::string embedType = parser.getCmdOption("-t");
    if (!embedType.empty()) param.embedType = embedType;
    else param.embedType = "Rademacher"; 
}

int main(int argc, char **argv){
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // root node generate random seed and broadcast it
    int seed;
    if (world_rank == 0) {
        seed = rand();
        std::cout << "seed: " << seed << std::endl; 
    }
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /* TODO: allow broadcasting the random matrix in case implementations are 
    different on nodes */

    Param param;
    parsing(argc, argv, param);
    DATATYPE X;
    arma::icolvec y;
    arma::icolvec sortedFlocs;
    load_train<DATATYPE>(X, y, param.trainDataPath, sortedFlocs);    

    //std::cout << "Data loaded." << std::endl;
    //std::cout << "Num data: " << X.n_rows << std::endl;
    //std::cout << "Num features: " << X.n_cols << std::endl;
    pdenParaWithEmbedding<DATATYPE, PARAMTYPE, arma::fmat> model(
        param.lambda1,
        param.lambda2
    );
    insertOne(X); 
    // keep data integradted for sparse matrix.
    // syncData(X) must be called before training.
    model.syncData(X);

    // assign work to each node
    int workload = param.embedDim / world_size;
    int reminder = param.embedDim % world_size;
    int startDim, endDim;
    if (world_rank < reminder) {
        startDim = (workload + 1) * world_rank;
        workload++;
    } 
    else{
        startDim = (workload + 1) * reminder + workload * (world_rank-reminder);
    }
    endDim = startDim + workload;
    std::cout << "Process " << world_rank << " handles from dim " << startDim;
    std::cout << " to " << endDim << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "Model training begins." << std::endl;
    }
    auto start = std::chrono::steady_clock::now();
    model.train(
        X,
        y,
        param.nIter,
        param.nPostIter,
        param.embedDim,
        startDim,
        endDim,
        param.embedType,
        sortedFlocs,
        seed
    );
    std::cout << "Process " << world_rank << " done training." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    
    if (world_rank == 0) {
        std::cout << "training time: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>
                     (end-start).count() / 1e6 << "s";
        std::cout << std::endl;
    }
    // collect trained model
    if (world_rank == 0) {
        const unsigned nFeat = X.n_cols;
        auto tempMat(model.weights);
        model.weights.zeros(nFeat, param.embedDim);
        memcpy(
            model.weights.memptr(),
            tempMat.memptr(),
            tempMat.n_elem * sizeof(float)
        );
        
        for (unsigned i = 1; i < world_size; ++i) {
            int iStartDim;
            int tempwl = param.embedDim / world_size;
            if (i < reminder) {
                iStartDim = (tempwl + 1) * i;
                tempwl++;
            } 
            else{
                iStartDim = (tempwl+1)*reminder + tempwl*(i-reminder);
            }
            MPI_Recv(
                model.weights.memptr() + nFeat * iStartDim,
                nFeat * tempwl,
                MPI_FLOAT,
                i,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        } 
    }
    else {
        MPI_Send(
            model.weights.memptr(),
            model.weights.n_elem,
            MPI_FLOAT,
            0,
            0,
            MPI_COMM_WORLD
        );
    }

    if (world_rank != 0) {
        MPI_Finalize();
        return 0;
    }
    start = end;
    end = std::chrono::steady_clock::now();
    
    std::cout << "collection time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>
        (end-start).count() / 1e6 << "s";
    std::cout << std::endl;
    // release the memory of training data:
    X.reset();
    y.reset();
    
    // inference
    DATATYPE XTest;
    arma::icolvec yTest;
    load_test<DATATYPE> (
        XTest,
        yTest,
        param.testDataPath,
        model.get_flocs(),
        model.get_feat_status()
    );
    std::cout << "Data loaded." << std::endl;
    std::cout << "Num data: " << XTest.n_rows << std::endl;
    std::cout << "Num features: " << XTest.n_cols << std::endl; 
    
    insertOne(XTest); 
    // keep data integradted for sparse matrix.
    // syncData(X) must be called before training.
    model.syncData(XTest);

    arma::icolvec preds;

    std::cout << "Inference begins." << std::endl; 
    start = std::chrono::steady_clock::now();
    model.predict(XTest, preds);
    end = std::chrono::steady_clock::now();
    std::cout << "Inferenece time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds> 
                 (end-start).count() / 1e6 << "s";
    std::cout << std::endl;
    
    double accuracy = arma::accu(preds == yTest) /(double) yTest.n_rows;
    std::cout << "Test accuracy: " <<  accuracy << std::endl;

    double l2error = model.computeRegressionError(XTest, yTest);
    std::cout << "l2 error on test data: " << l2error << std::endl;
    MPI_Finalize();
    return 0;
}
