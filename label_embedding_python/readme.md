# PyTorch Neural Network Implementation README

This README provides essential instructions for setting up the environment, understanding the code structure, and running our PyTorch implementation of neural networks.

## Environment Setup

The environment setup is stored in the `env.yml` file. 

## Code Structure

Our codebase includes various Python scripts, with each file serving a different purpose. Here's a brief rundown:

- `nn_utils.py`: This file contains utility functions used in our neural network models.
- `run_nn_[dataset].py`: These scripts implement the embedding methods for different datasets. Replace `[dataset]` with the name of the dataset you are working with (e.g., `lshtc1`, `dmoz`, `odp`).
- `run_nn_ce_[dataset].py` and `run_nn_sq_[dataset].py`: These files implement the cross entropy and squared loss baselines for different datasets, respectively. 

## Running the Code

To run the embedding framework scripts, use the following command:

```bash
python run_nn_[dataset].py <embedding dimension>
```

Replace `[dataset]` with the name of the dataset you are working with, and `<embedding dimension>` with the desired embedding dimension. For example:

```bash
python run_nn_lshtc1.py 512
```

To run the baseline scripts for cross entropy loss and squared loss, use the respective command:

```bash
python run_nn_ce_[dataset].py
python run_nn_sq_[dataset].py
```

These scripts don't require any arguments to run.

## Data

Our data can be downloaded from this link: [https://drive.google.com/drive/folders/1K3lU4vhHTFYrZs2L5sY9uC3gRGeft5aS?usp=sharing](https://drive.google.com/drive/folders/1K3lU4vhHTFYrZs2L5sY9uC3gRGeft5aS?usp=sharing).
