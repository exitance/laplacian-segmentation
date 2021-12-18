# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os
import gc

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def get_device():
    '''Get device (if GPU is available, use GPU)'''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# calculate curvature for polyline
# k[j] = 2 * \alpha[j] / (l1[j] + l2[j])
def curvature(eigv):
    curv = np.zeros((eigv.shape[0], eigv.shape[1]-2))
    for i in range(eigv.shape[0]):
        # k--the slope of this line segment: (eigv[j] - eigv[j-1]) / (j-(j-1))
        # beta--the angle between this line segment and the x-axis: arctan(k)
        beta = np.arctan(eigv[i][1] - eigv[i][0])
        # len1--the length of this line segment (j, eigv[j]), (j-1, eigv[j-1]) 
        len1 = np.sqrt(pow(eigv[i][1] - eigv[i][0], 2) + 1)
        for j in range(1, eigv.shape[1]-1):
            betaNext = np.arctan(eigv[i][j+1] - eigv[i][j])
            # the Normal turning angle \alph
            # 
            alpha = beta - betaNext
            len2 = np.sqrt(pow(eigv[i][j+1] - eigv[i][j], 2) + 1)
            curv[i][j-1] = 2 * alpha / (len1 + len2)
            beta = betaNext
            len1 = len2
    return curv

class EigenvalueDataset(Dataset):
    ''' Dataset for loading and preprocessing the eigenvalue dataset '''
    def __init__(self,
                 path,
                 mode='train'):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data).astype(float)
            lable = np.array(data[:, -1]).astype(int)
            data = np.array(data[:, 0:data.shape[1]-1]).astype(float)
            data[:, 0] = 0 # every first eigenvalue is 0
            data = curvature(data)
            self.data = torch.FloatTensor(data)
            self.label = torch.LongTensor(lable)
            self.dim = self.data.shape[1]

        print('Finished reading the {} set of Eigenvalue Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

def prep_dataloader(path, mode, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = EigenvalueDataset(path, mode=mode)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)        # Construct dataloader
    return dataloader

# Model
class Segmentation(nn.Module):
    def __init__(self, dim):
        '''A simple fully-connected deep neural network'''
        super(Segmentation, self).__init__()
        self.layer1 = nn.Linear(dim, dim+10)
        self.out = nn.Linear(dim+10, dim)

        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        '''Given input of size (batch_size x input_dim), compute output of the network'''
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.out(x)

        return x

# training method
def train(train_loader, test_loader, model, num_epoch, criterion, optimizer, model_path, device):
    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs) 
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1)   # get the index of the class with the highest probability
            batch_loss.backward()
            optimizer.step() 

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # testing
        if len(test_loader) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels) 
                    _, test_pred = torch.max(outputs, 1)
                
                    test_acc += (test_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    test_loss += batch_loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc/len(train_loader.dataset), train_loss/len(train_loader.dataset), test_acc/len(test_loader.dataset), test_loss/len(test_loader.dataset)
                ))

                # if the model improves, save a checkpoint at this epoch
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc/len(test_loader.dataset)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_loader.dataset), train_loss/len(train_loader.dataset)
            ))

    # if not validating, save the last epoch
    if len(test_loader) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')

# make predict
def predict(test_loader, model, device):
    pred = []
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

            for y in test_pred.cpu().numpy():
                pred.append(y)
    return pred

# configures
BATCH_SIZE = 20
VAL_RATIO = 0.2
# get device
device = get_device()
print(f'DEVICE: {device}')
# training parameters
num_epoch = 20          # number of training epoch
learning_rate = 0.0001  # learning rate

os.makedirs('models', exist_ok=True)

dataset_root = 'dataset/eig'
train_file = 'train.csv'
test_file = 'test.csv'
model_root = 'models'
model_file = 'model.pth'

dirs = ['coseg_aliens', 'coseg_chairs', 'coseg_vases']

for dir in dirs:
    print('model for {0}:\n'.format(dir))
    print('Loading data ...\n')
    # the path where data loaded
    dataset_path = dataset_root + '/' + dir
    train_path = dataset_path + '/' + train_file
    test_path = dataset_path + '/' + test_file
    # load dataset
    train_loader = prep_dataloader(train_path, 'train', BATCH_SIZE)
    test_loader = prep_dataloader(test_path, 'test', BATCH_SIZE)
    # the path where checkpoint saved
    model_path = model_root + '/' + dir + '/' + model_file
    os.makedirs(model_root + '/' + dir, exist_ok=True)    # The trained model will be saved to ./models/<dir>
    # training
    # create model, define a loss function, and optimizer
    model = Segmentation(train_loader.dataset.dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('Start training ...\n')
    train(train_loader, test_loader, model, num_epoch, criterion, optimizer, model_path, device)
    # create model and load weight from checkpoint to make predicts
    # ...