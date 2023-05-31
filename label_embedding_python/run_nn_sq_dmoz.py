import torch
import numpy as np
from time import time
from sklearn.datasets import load_svmlight_files 
import math
from nn_utils import *

import sys
seed = 20220510  # gonna use this integer to sample random seeds for different functions
max_int = np.iinfo(np.int32).max
rng = np.random.default_rng(seed)

train_path = # TODO: fill the data path
val_path = # TODO: fill the data path
test_path = # TODO: fill the data path
X_train, y_train, X_val, y_val, X_test, y_test = load_svmlight_files((train_path, val_path, test_path), dtype=np.float32, multilabel=False)

print("Neural neworks baseline on Dmoz")
print("num of training data: ", X_train.shape[0])
print("num of validation data: ", X_val.shape[0])
print("num of test data: ", X_test.shape[0])
print("num of features: ", X_train.shape[1])

num_classes = len(set(y_train.tolist() + y_val.tolist() + y_test.tolist()))
print("num of classes: ", num_classes)

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)

num_features = X_train.shape[1]

class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.model(x)

train_loader = torch.utils.data.DataLoader(sparse_dataset(X_train, y_train), batch_size=256, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)
test_loader = torch.utils.data.DataLoader(sparse_dataset(X_test, y_test), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)
val_loader = torch.utils.data.DataLoader(sparse_dataset(X_val, y_val), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)

device = torch.device("cuda")

epochs = 5
loss_f = SquaredLoss()
model = Net(num_features, 2500, num_classes).to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=0e-6)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, ], gamma=0.1)

epoch_time_hist = [0 ,]
train_time = 0
val_loss_hist = []
val_acc_hist = []
test_loss_hist = []
test_acc_hist = []
val_loss, val_acc = test_ce(model, loss_f, device, val_loader)
print("before training. validation results. l2_loss: {:.6f}, accuracy: {:.4f}".format(val_loss, val_acc))
test_loss, test_acc = test_ce(model, loss_f, device, test_loader)
print("before training. test results. l2_loss: {:.6f}, accuracy: {:.4f}".format(test_loss, test_acc))

for epoch in range(1, epochs+1):
    start = time()
    train_ce(model, loss_f, device, train_loader, optimizer, epoch)
    scheduler.step()
    train_time += time() - start
    val_loss, val_acc = test_ce(model, loss_f, device, val_loader)
    print("validation results. l2_loss: {:.6f}, accuracy: {:.4f}".format(val_loss, val_acc))
    test_loss, test_acc = test_ce(model, loss_f, device, test_loader)
    print("test results. l2_loss: {:.6f}, accuracy: {:.4f}".format(test_loss, test_acc))
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)
    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)
    epoch_time_hist.append(train_time)

# measure prediction time:
prediction_start = time()
test_loss, test_acc = test_ce(model, loss_f, device, test_loader)
prediction_time = time() - prediction_start

print("validation loss: ", val_loss_hist)
print("validation accuracy: ", val_acc_hist)
print("test loss: ", test_loss_hist)
print("test accuracy: ", test_acc_hist)
print("training time by epoch = ", epoch_time_hist)
print("prediction time = ", prediction_time)