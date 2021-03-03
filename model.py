import torch
import torch.nn as nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    """ A implementation of the GoogLeNet architecture from the paper '' """
    def __init__(self, n_class):
        super().__init__()
        self.training = True
        
        # Layers 1 to 4
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, padding_mode='reflect') 
        self.redu1 = nn.Conv2d(64,64,kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        
        # Layers 5 to 8
        self.incept1 = Inception(192,64,(96,128),(16,32),32)
        self.incept2 = Inception(256,128,(128,192),(32,96),64)
        self.incept3 = Inception(480,192,(96,208),(16,48),64)
        
        # Optional Classifcation Path 1
        self.redu2 = nn.Conv2d(512,128, kernel_size=1,stride=1)
        self.fc1 = nn.Linear(128*4*4,1024)
        self.fc2 = nn.Linear(1024, n_class)
        
        # Layers 9 to 11
        self.incept4 = Inception(512,160,(112,224),(24,64),64)
        self.incept5 = Inception(512,128,(128,256),(24,64),64)
        self.incept6 = Inception(512,112,(144,288),(32,64),64)
        
        # Optional Classifcation Path 2
        self.redu3 = nn.Conv2d(528,128, kernel_size=1,stride=1)
        self.fc3 = nn.Linear(128*4*4,1024)
        self.fc4 = nn.Linear(1024, n_class) 
        
        # Layers 12 to 15
        self.incept7 = Inception(528,256,(160,320),(32,128),128)
        self.incept8 = Inception(832,256,(160,320),(32,128),128)
        self.incept9 = Inception(832,384,(192,384),(48,128),128)
        
        # Final Classifcation Path
        self.fc5 = nn.Linear(1024,n_class)
        
        # Unparamartried (spelling?) layers
        self.avg_pool1 = nn.AvgPool2d(5, stride=3)
        self.avg_pool2 = nn.AvgPool2d(7, stride=1)
        self.max_pool = nn.MaxPool2d(3,stride=2,padding=1)
        self.drop1 = nn.Dropout(0.7)
        self.drop2 = nn.Dropout(0.4)
        self.lrn = nn.LocalResponseNorm(size=5,alpha=10e-4,beta=0.75,k=2.0)
        

    def forward(self, x):
        # Layers 1 to 4
        x = F.relu(self.conv1(x))
        x = self.lrn(self.max_pool(x))
        x = F.relu(self.redu1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(self.lrn(x))
        # Layers 5 to 8
        x = self.incept1(x)
        x = self.incept2(x)
        x = self.max_pool(x)
        x = self.incept3(x)
        if self.training:
            # Optional Classifcation Path 1
            x1 = self.avg_pool1(x)
            x1 = F.relu(self.redu2(x1))
            x1 = x1.view(-1,128*4*4)
            x1 = F.relu(self.fc1(x1))
            x1 = F.relu(self.fc2(x1))
            # softmax?
        # Layers 9 to 11
        x = self.incept4(x)
        x = self.incept5(x)
        x = self.incept6(x)
        if self.training:
            # Optional Classifcation Path 2
            x2 = self.avg_pool1(x)
            x2 = F.relu(self.redu3(x2))
            x2 = x2.view(-1,128*4*4)
            x2 = F.relu(self.fc3(x2))
            x2 = F.relu(self.fc4(x2))
            # softmax?
        # Layers 12 to 15
        x = self.incept7(x)
        x = self.max_pool(x)
        x = self.incept8(x)
        x = self.incept9(x)
        # Final Classifcation Path
        x = self.avg_pool2(x)
        x = x.view(-1,1024)
        x = self.fc5(x)
        # softmax?
        if self.training:
            return x1, x2, x
        else:
            return x

class Inception(nn.Module):
    """ An implementation of the Inception blocks desired in the GoogLeNet Paper """
    def __init__(self, dim_in, dim_1, dims_3, dims_5, dim_pool):
        super().__init__()
        # 1x1 Convolution Path
        self.redu1 = nn.Conv2d(dim_in, dim_1, kernel_size=1, stride=1) 
        # 3x3 Convolution Path
        self.redu2 = nn.Conv2d(dim_in, dims_3[0], kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(dims_3[0], dims_3[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # 5x5 Convolution Path
        self.redu3 = nn.Conv2d(dim_in, dims_5[0], kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dims_5[0], dims_5[1], kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        # Max-Pool Pat
        self.pool  = nn.MaxPool2d(3,1, padding=1)
        self.redu4 = nn.Conv2d(dim_in, dim_pool, kernel_size=1,stride=1)

    def forward(self,x):
        # 1x1 Convolution Path
        x1 = F.relu(self.redu1(x))
        
        # 3x3 Convolution Path
        x2 = F.relu(self.redu2(x))
        x2 = F.relu(self.conv1(x2))
        # 5x5 Convolution Path
        x3 = F.relu(self.redu3(x))
        x3 = F.relu(self.conv2(x3))
        
        # Max-Pool Path
        x4 = self.pool(x)
        x4 = F.relu(self.redu4(x4))
        
        #  Concatenate the resulting tensors
        x = torch.cat((x1,x2,x3,x4),1) 
        return x
