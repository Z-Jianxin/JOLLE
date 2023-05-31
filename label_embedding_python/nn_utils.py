import torch
from sklearn.metrics import pairwise_distances
import numpy as np

class sparse_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_features = x.shape[1]
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        return self.x.indices[self.x.indptr[i]:self.x.indptr[i+1]], self.x.data[self.x.indptr[i]:self.x.indptr[i+1]], self.y[i], self.n_features 

def sparse_collate_coo(batch):
    r = []
    c = []
    vals = []
    y = []
    n_features = batch[0][-1]
    for i, (indices, data, yi, _) in enumerate(batch):
        r.extend([i] * indices.shape[0])
        c.extend(indices)
        vals.extend(data)
        y.append(yi)
    return ([r, c], vals, (len(batch), n_features)), y

class SquaredLoss(torch.nn.Module):
    def __init__(self):
        super(SquaredLoss, self).__init__()

    def forward(self, outputs, targets):
        one_hot_approx = torch.zeros_like(outputs)
        one_hot_approx.scatter_(1, targets.unsqueeze(1), 1)
        return torch.sum((outputs - one_hot_approx) ** 2)

def train_le(model, label_embed, loss_f, device, train_loader, optimizer, epoch, log_interval=50):
    model.train()
    for idx, ((locs, vals, size), y) in enumerate(train_loader):
        x = torch.sparse_coo_tensor(locs, vals, size=size, dtype=torch.float32, device=device)
        y_embed = torch.index_select(label_embed, 0, torch.tensor(y, dtype=torch.int32).to(device))
        optimizer.zero_grad()
        embed_out = model(x)
        loss = loss_f(embed_out, y_embed) / len(y)
        loss.backward()
        optimizer.step()
        if (idx + 1) % log_interval == 0:
            print("train epoch: {}, batch: {}/{}, loss: {:.6f}".format(epoch, idx+1, len(train_loader), loss.item()))

def find1NN_cuda(out_cuda, label_embed_cuda):
    #dist_m = torch.cdist(out_cuda.reshape(1, out_cuda.shape[0], -1), label_embed_cuda.reshape(1, label_embed_cuda.shape[0], -1))
    #dist_m = dist_m.reshape(dist_m.shape[1], -1)
    #oneNNs = torch.argmin(dist_m, dim=1)
    gram_m = torch.matmul(out_cuda, torch.transpose(label_embed_cuda, 0, 1))
    return torch.argmax(gram_m, dim=1)
            
def test_le(model, label_embed, loss_f, device, test_loader):
    model.eval()
    mean_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, ((locs, vals, size), y) in enumerate(test_loader):
            x = torch.sparse_coo_tensor(locs, vals, size=size, dtype=torch.float32, device=device)
            y_embed = torch.index_select(label_embed, 0, torch.tensor(y, dtype=torch.int32).to(device))
            embed_out = model(x)
            mean_loss += loss_f(embed_out, y_embed).item()
            embed_out_detached = embed_out.detach()
            preds = find1NN_cuda(embed_out_detached, label_embed).cpu().numpy()
            correct += np.sum(preds==y)
            total += preds.shape[0]
            del x, y_embed, embed_out
    return mean_loss / len(test_loader.dataset), correct/total

def train_ce(model, loss_f, device, train_loader, optimizer, epoch, log_interval=50):
    model.train()
    for idx, ((locs, vals, size), y) in enumerate(train_loader):
        x = torch.sparse_coo_tensor(locs, vals, size=size, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_f(out, torch.tensor(y, dtype=torch.int64).to(device)) / len(y)
        loss.backward()
        optimizer.step()
        if (idx + 1) % 10 == 0:
            print("train epoch: {}, batch: {}/{}, loss: {:.6f}".format(epoch, idx+1, len(train_loader), loss.item()))

def test_ce(model, loss_f, device, test_loader):
    model.eval()
    mean_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, ((locs, vals, size), y) in enumerate(test_loader):
            x = torch.sparse_coo_tensor(locs, vals, size=size, dtype=torch.float32, device=device)
            out = model(x)
            mean_loss += loss_f(out, torch.tensor(y, dtype=torch.int64).to(device)).item()
            preds = out.detach().cpu().argmax(dim=1, keepdim=False).numpy()
            correct += np.sum(preds==np.array(y))
            total += preds.shape[0]
    return mean_loss / len(test_loader.dataset), correct/total