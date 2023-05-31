# C++ Experiment Code README

This README provides essential instructions on how to install dependencies, compile, and run our C++ experimental code. This project includes two key components: an Elastic Net implementation and a Random Forest implementation. Our data can be downloaded from this link: [https://drive.google.com/drive/folders/1K3lU4vhHTFYrZs2L5sY9uC3gRGeft5aS?usp=sharing](https://drive.google.com/drive/folders/1K3lU4vhHTFYrZs2L5sY9uC3gRGeft5aS?usp=sharing).

## Dependencies

Our code relies on the Armadillo linear algebra package. We recommend using [Conda](https://conda.io/), a package and environment management system, to handle the installation. 

Please note that different choices of underlying BLAS and LAPACK routines may result in performance variation compared to those reported in our paper. Our environment setup is stored in the file `./env.yml`.

To install Armadillo using Conda, use the following command:

```bash
conda install -c conda-forge armadillo
```

This command will install Armadillo in your active Conda environment. Note that the default Blas and Lapack packages installed with the above command may not be the optimal choice for your machine. The assumption here is that Armadillo is installed in an environment named "arma".

## Compilation

### Single-Node Elastic Net

To compile the single-node Elastic Net, use the following command:

```bash
g++ -O3 -std=c++11 -fopenmp en.cpp -I${HOME}/anaconda3/envs/arma/include \
    -Wl,-rpath=${HOME}/anaconda3/envs/arma/lib \
    -L${HOME}/anaconda3/envs/arma/lib \
    -larmadillo -DSPARSE_DATA -DARMA_NO_DEBUG -o en
```

### Single-Node Random Forest

To compile the single-node Random Forest, use the following command:

```bash
g++ -O3 -std=c++11 -fopenmp rf.cpp \
    -I${HOME}/anaconda3/envs/arma/include \
    -Wl,-rpath=${HOME}/anaconda3/envs/arma/lib \
    -L${HOME}/anaconda3/envs/arma/lib \
    -larmadillo -o rf -DARMA_NO_DEBUG -DSPARSE_DATA
```

## Execution

To run the Elastic Net, use the following command:

```bash
./en -trd <train_data_path> -ted <test_data_path> \
    -e <embedding dimension> -n <solver iterations> \
    -p 0<post iterations, keep it to be 0 please> \
    -l1 <l1_regularization> -l2 <l2_regularization>
```

To run the Random Forest, use the following command:

```bash
./rf -trd <train_data_path> -ted <test_data_path> \
    -e <embedding dimension> -mf <max number of features to check in a split> \
    -md <max_depth> -ml <minimal leaf size> -ms <minimal sample size for a split> \
    -sc 1e-9 <numerical tolerance, keep it as 1e-9 please> -n <number of trees>
```

## Distributed Programs

For distributed programs, we use the Intel MPI runtime in the Slurm cluster. Load the module by using the following command:

```bash
module load impi/2021.5.1
```

### Compilation

For Elastic Net:

```bash
mpicxx -O3 -std=c++11 -fopenmp enMPI.cpp \
    -I${HOME}/anaconda3/envs/arma/include \
    -Wl,-rpath=${HOME}/anaconda3/envs/arma/lib \
    -L${HOME}/anaconda3/envs/arma/lib \
    -larmadillo -o enMPI -DSPARSE_DATA -DARMA_NO_DEBUG
```

For Random Forest:

```bash
mpicxx -O3 -std=c++11 -fopenmp rfMPI.cpp \
    -I${HOME}/anaconda3/envs/arma/include \
    -Wl,-rpath=${HOME}/anaconda3/envs/arma/lib \
    -L${HOME}/anaconda3/envs/arma/lib \
    -larmadillo -o rfMPI -DSPARSE_DATA -DARMA_NO_DEBUG
```

### Execution

To run Elastic Net:

```bash
mpiexec ./enMPI -trd <train_data_path> -ted <test_data_path> \
    -e <embedding dimension> -n <solver iterations> -p 0<post iterations, keep it to be 0 please> \
    -l1 <l1_regularization> -l2 <l2_regularization>
```

To run Random Forest:

```bash
mpiexec ./rfMPI -trd <train_data_path> -ted <test_data_path> \
    -e <embedding dimension> -mf <max number of features to check in a split> \
    -md <max_depth> -ml <minimal leaf size> -ms <minimal sample size for a split> \
    -sc 1e-9 <numerical tolerance, keep it as 1e-9 please> -n <number of trees>
```

Please note that the Intel MPI (impi) package is needed for both compilation and execution of the distributed programs.