import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pickle
from model import GoogLeNet 

# Select device to train on
device = torch.device("cuda")

def train(model, train_dl, test_dl, opt, loss_func, epochs):
    """ train model using using provided datasets, optimizer and loss function """
    train_loss = [0 for i in range(epochs)]
    test_loss = [0 for i in range(epochs)]
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_func(model(xb), yb)
            train_loss[epoch] = loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
        print("Finished the epoch! Any errors from here on are to do with the test vs train modes") 
        with torch.no_grad():
            losses, nums = zip(*[(loss_func(model(xb.to(device)),yb.to(device)).item(),len(xb.to(device))) for xb, yb in test_dl])
            test_loss[epoch] = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            correct = 0
            total = 0
            
            for data in test_dl:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss[epoch]}, Test Loss {test_loss[epoch]}, Accuracy: {100*correct/total}')
    
    return train_loss, test_loss

if __name__ == "__main__":
    # Define transforms
    transform = transforms.Compose([torchvision.transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load training data
    dataset = datasets.ImageFolder('./data', transform=transform) 
    test_data, train_data = random_split(dataset,(4396,1099), generator=torch.Generator().manual_seed(42))
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)
    
    # Train Model
    # Some modification will be required here due to the somewhat strange method of training for googlenet
    epochs = 25
    model = GoogLeNet(11)
    loss_func = lambda x, y : 0.3*nn.CrossEntropyLoss(x[0],y)+0.3*nn.CrossEntropyLoss(x[1],y)+nn.CrossEntropyLoss(x[2],y) 
    opt = optim.Adam(model.parameters(), lr=0.0001)
    train_loss, test_loss = train(model, train_dl, test_dl, opt, loss_func, epochs)
    
    # Save Model to pkl file
    f = open(f'model.pkl','wb')
    pickle.dump(model,f)
    f.close()
