import torch
import numpy as np
from time import time
from sklearn.datasets import load_svmlight_files 
import math
from nn_utils import *

import sys

seed = 20230508
max_int = np.iinfo(np.int32).max
rng = np.random.default_rng(seed)

train_path = # TODO: fill the data path
val_path = # TODO: fill the data path
test_path = # TODO: fill the data path
X_train, y_train, X_val, y_val, X_test, y_test = load_svmlight_files((train_path, val_path, test_path), dtype=np.float32, multilabel=False)

print("Experiments on Dmoz")
print("num of training data: ", X_train.shape[0])
print("num of validation data: ", X_val.shape[0])
print("num of test data: ", X_test.shape[0])
print("num of features: ", X_train.shape[1])

num_classes = len(set(y_train.tolist() + y_val.tolist() + y_test.tolist()))
y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)

num_features = X_train.shape[1]

class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, embed_dim):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        x = self.model(x)
        row_norms = torch.norm(x, dim=1, p=2).unsqueeze(1)
        return x/row_norms

train_loader = torch.utils.data.DataLoader(sparse_dataset(X_train, y_train), batch_size=256, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)
test_loader = torch.utils.data.DataLoader(sparse_dataset(X_test, y_test), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)
val_loader = torch.utils.data.DataLoader(sparse_dataset(X_val, y_val), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)

device = torch.device("cuda")

def run_exp(embed_dim):
    label_embed = (rng.integers(low=0, high=2, size=(num_classes, embed_dim)) * 2 - 1) / math.sqrt(embed_dim)
    epsilon = np.max(np.abs(label_embed @ label_embed.T - np.eye(num_classes)))
    print("epsilon = ", epsilon)
    label_embed = torch.tensor(np.float32(label_embed)).to(device)

    epochs = 5
    loss_f = torch.nn.MSELoss(reduction='sum')
    model = Net(num_features=num_features, hidden_dim=2500, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=0e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)

    epoch_time_hist = []
    train_time = 0
    val_loss_hist = []
    val_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in range(1, epochs+1):
        start = time()
        train_le(model, label_embed, loss_f, device, train_loader, optimizer, epoch)
        scheduler.step()
        train_time += time() - start
        val_loss, val_acc = test_le(model, label_embed, loss_f, device, val_loader)
        print("validation results. l2_loss: {:.6f}, accuracy: {:.4f}".format(val_loss, val_acc))
        test_loss, test_acc = test_le(model, label_embed, loss_f, device, test_loader)
        print("test results. l2_loss: {:.6f}, accuracy: {:.4f}".format(test_loss, test_acc))
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        epoch_time_hist.append(train_time)

    # measure prediction time:
    prediction_start = time()
    test_loss, test_acc = test_le(model, label_embed, loss_f, device, test_loader)
    prediction_time = time() - prediction_start
    return val_loss_hist, val_acc_hist, test_loss_hist, test_acc_hist, epoch_time_hist, prediction_time

val_loss_hist, val_acc_hist, test_loss_hist, test_acc_hist, epoch_time_hist, prediction_time = run_exp(int(sys.argv[1]))
print("validation loss: ", val_loss_hist)
print("validation accuracy: ", val_acc_hist)
print("test loss: ", test_loss_hist)
print("test accuracy: ", test_acc_hist)
print("training time by epoch = ", epoch_time_hist)
print("prediction time = ", prediction_time)