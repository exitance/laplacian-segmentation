import numpy as np
from numpy.core.fromnumeric import searchsorted
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import gc

# Hyperparameters and Constants
# training parameters
n_epochs = 20
learning_rate = 1e-4

batch_size = 64
val_ratio = 0.2
seed = 42
input_dim = 6
# the path where checkpoint saved
model_path = ''

# check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# calculate curvature
def curvature(data_eig, data):
    for i in range(data_eig.shape[0]):
        alpha = m.atan(data_eig[i][1] - data_eig[i][0])
        len1 = m.sqrt(pow(data_eig[i][1] - data_eig[i][0], 2) + 4)
        for j in range(1, data_eig.shape[1]-1):
            k = data_eig[i][j+1] - data_eig[i][j]
            a = m.atan(k)
            alp = alpha - a
            len2 = m.sqrt(pow(data_eig[i][j+1] - data_eig[i][j], 2) + 4)
            train[i][j-1] = 2 * alp / (len1 + len2)
            alpha = a
            len1 = len2

# Preparing Data
print('Loading data ...')

data_root=''

import math as m

train_eig = np.load(data_root + '')
train_eig.argsort() # eigenvalues in ascending order
input_dim = train_eig.shape[1]
train = np.zero((train_eig.shape[0], input_dim), dtype = np.float)
curvature(train_eig, train)
train_label = np.load(data_root + '')

test_eig = np.load(data_root + '')
test_eig.argsort()
test = np.zero((test_eig.shape[0], input_dim), dtype = np.float)
curvature(test_eig, test)

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))

# Dataset
class LaplacianDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

percent = int(train.shape[0] * (1 - val_ratio))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

train_set = LaplacianDataset(train_x, train_y)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True) # only shuffle training data
val_set = LaplacianDataset(train_x, train_y)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()

# Create Model
class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.layer1 = nn.Linear(input_dim, )
        self.layer2 = nn.Linear()
        self.layer3 = nn.Linear()
        self.out = nn.Linear()

        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)
                
        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

# Training

# fix random seed for reproducibility
same_seeds(seed)

# get device
device = get_device()
print(f'DEVICE: {device}')

model = Segmentation().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# start training

best_acc = 0.0
for epoch in range(n_epochs):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    # training
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs += 1
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs += 1
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += batch_loss.item()
            
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, n_epochs, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, n_epochs, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

# Testing
# create testing data
test_set = LaplacianDataset(test, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# create model and load weights from checkpoint
model = Segmentation().to(device)
model.load_state_dict(torch.load(model_path))

# make prediction
predict = []
model.eval() # set the model to evaluation mode

# Plot
"""
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(train_loss[:])
plt.title("train_loss")
plt.subplot(122)
plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
plt.title("epochs_loss")
plt.legend()
plt.show()
"""