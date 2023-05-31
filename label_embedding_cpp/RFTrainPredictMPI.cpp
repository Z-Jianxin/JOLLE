#include <mpi.h>
#include "rf.h"
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
    // parameter for the forest
    int nEstimators;
    // parameters for trees:
    int minSamplesSplit;  
    int minSamplesLeaf;
    int maxFeatures;
    int maxDepth;
    double stopCriterion;
    long int seed;
};

void parsing(int argc, char** argv, Param & param){
    InputParser parser(argc, argv);
    if(parser.cmdOptionExists("-h")){
        std::cerr << "output some help information" << std::endl;
        exit(0);
    }
    
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
    
    const std::string embedDimStr = parser.getCmdOption("-e");
    if (!embedDimStr.empty()) param.embedDim = std::stoi(embedDimStr);
    else{
        std::cerr << "embedding dimension unspecified" << std::endl;
        exit(1);
    }

    const std::string embedType = parser.getCmdOption("-t");
    if (!embedType.empty()) param.embedType = embedType;
    else param.embedType = "Rademacher";

    const std::string nEstStr = parser.getCmdOption("-n");
    if (!nEstStr.empty()) param.nEstimators = std::stoi(nEstStr);
    else param.nEstimators = 20; 

    const std::string msStr = parser.getCmdOption("-ms");
    if (!msStr.empty()) param.minSamplesSplit = std::stoi(msStr);
    else param.minSamplesSplit = 20;

    const std::string mlStr = parser.getCmdOption("-ml");
    if (!mlStr.empty()) param.minSamplesLeaf = std::stoi(mlStr);
    else param.minSamplesLeaf = 10;

    const std::string mfStr = parser.getCmdOption("-mf");
    if (!mfStr.empty()) param.maxFeatures = std::stoi(mfStr);
    else param.maxFeatures = 1000;

    const std::string mdStr = parser.getCmdOption("-md");
    if (!mdStr.empty()) param.maxDepth = std::stoi(mdStr);
    else param.maxDepth = 64;

    const std::string scStr = parser.getCmdOption("-sc");
    if (!scStr.empty()) param.stopCriterion = std::stod(scStr);
    else param.stopCriterion = 1e-6;

    const std::string seedStr = parser.getCmdOption("-s");
    if (!seedStr.empty()) param.seed = std::stol(seedStr);
    else param.seed = -1;
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

    // root node generate random seed for embedding matrix and broadcast it
    int embed_seed;
    if (world_rank == 0) {
        embed_seed = rand();
        std::cout << "seed for label embedding: " << embed_seed << std::endl; 
    }
    MPI_Bcast(&embed_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /* TODO: allow broadcasting the random matrix in case implementations are 
    different on nodes */
   
    Param param;
    parsing(argc, argv, param);
    DATATYPE X;
    arma::icolvec y;
    arma::icolvec sortedFlocs;
    load_train<DATATYPE>(X, y, param.trainDataPath, sortedFlocs);    
        unsigned seed;
    if (param.seed == -1) seed = 
        (unsigned) std::chrono::system_clock::now().time_since_epoch().count();
    else seed = (unsigned) param.seed;

    std::cout << "seed: " << seed << " on node: " << world_rank << std::endl;

    if (world_rank == 0) { 
        std::cout << "Data loaded." << std::endl;
        std::cout << "Num data: " << X.n_rows << std::endl;
        std::cout << "Num features: " << X.n_cols << std::endl;
    }

    // assign workload
    int workload = param.nEstimators / world_size;
    int reminder =  param.nEstimators % world_size;
    if (world_rank < reminder) ++workload;
    
    RFWithEmbedding<DATATYPE, arma::mat> model(
        workload,
        param.minSamplesSplit,
        param.minSamplesLeaf,
        param.maxFeatures,
        param.maxDepth,
        param.stopCriterion,
        seed
    );
    
    // keep data integradted for sparse matrix.
    // syncData(X) must be called before training.
    model.syncData(X);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "Model training begins." << std::endl;
    }
    auto start = std::chrono::steady_clock::now();
    model.train(
        X,
        y,
        param.embedDim,
        param.embedType,
        sortedFlocs,
        embed_seed
    );
    std::cout << "Process " << world_rank << " done training." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    if (world_rank == 0) {
        std::cout << "Training completes." << std::endl;
        std::cout << "Training time: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>
            (end-start).count() / 1e6 << "s";
        std::cout << std::endl;
    }
    
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
    if (world_rank == 0){
        std::cout << "Data loaded." << std::endl;
        std::cout << "Num data: " << XTest.n_rows << std::endl;
        std::cout << "Num features: " << XTest.n_cols << std::endl;
    }
    insertOne(XTest); 
    // keep data integradted for sparse matrix.
    // syncData(X) must be called before training.
    model.syncData(XTest);

    arma::mat predEmbed;

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "Inference begins." << std::endl; 
        start = std::chrono::steady_clock::now();
    }
    model.predictEmbed(XTest, predEmbed);
    predEmbed = predEmbed * model.nEstimators;
    if (world_rank == 0) {
        arma::mat temp(XTest.n_rows, param.embedDim, arma::fill::zeros);
        for (unsigned i = 1; i < world_size; ++i){
            MPI_Recv(
                temp.memptr(),
                temp.n_elem,
                MPI_DOUBLE,
                i,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
            predEmbed += temp;
        }  
    } else {
        MPI_Send(
            predEmbed.memptr(),
            predEmbed.n_elem,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD
        );
        MPI_Finalize();
        return 0;
    }
    if (world_rank == 0) { 
        end = std::chrono::steady_clock::now();
        std::cout << "Inferenece time: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds> 
            (end-start).count() / 1e6 << "s";
        std::cout << std::endl; 
    }
    predEmbed /= param.nEstimators; 
    
    arma::icolvec preds;
    model.PredictHelper(predEmbed, preds);
    
    double accuracy = arma::accu(preds == yTest) /(double) yTest.n_rows;
    std::cout << "Test accuracy: " <<  accuracy << std::endl;
    MPI_Finalize();
    return 0;
}
