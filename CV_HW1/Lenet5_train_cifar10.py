############################################################################
#                   Cv.hw1 (train cifar10 with lenet5.)                    #  
#                        Arthor: Wet-ting Cao.                             #   
#                             2019.10.24                                   #
############################################################################

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import matplotlib.pyplot as plt

epochs = 100
batch = 256
lr = 0.01
optimizer = 'SGD'
    
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                          
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=2)                                          

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        
        # (3, 32, 32) -> (6, 14, 14) -> (16, 5, 5)
        self.conv = nn.Sequential( 
            nn.Conv2d(3, 6, 5, 1, 0), # in_ch, out_ch, kernel, stride, padding.
            nn.AvgPool2d(2, 2, 0),  # kernel, stride, padding.
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.AvgPool2d(2, 2, 0)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(), 
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv(x)        
        x = x.view(batchsize, 16 * 5 * 5)
        x = self.fc(x) # logits
        return x
        
def train():
    device = torch.device("cuda")
    model = Lenet5().to(device)
    criteron = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)
    loss_epoch = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        iter = 0
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0
        train_loss = 0.0
        model.train()
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))
        
        for i, (x, label) in enumerate(trainloader) :
            x, label = x.to(device), label.to(device)
            y = model(x)
            loss = criteron(y, label)      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
            train_loss += loss.item()
            iter += 1
        #print(train_loss, iter, batch)    
        print('Training: epoch: %d / loss: %.3f / acc: %.3f '  % (epoch + 1, train_loss / iter, correct_train / total_train))
        
        model.eval()
        for i, (x, label) in enumerate(testloader) :
            with torch.no_grad():
                x, label = x.to(device), label.to(device)
                y = model(x)
                _, predicted = torch.max(y, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum().item()
        print('Testing: epoch: %d / acc: %.3f '  % (epoch + 1, correct_test / total_test))
        #print(correct_train, total_train, correct_test, total_test)
        
        train_acc.append(100 * correct_train / total_train)
        test_acc.append(100 * correct_test / total_test)        
        loss_epoch.append(train_loss / iter)
    
        plt.figure()
        plt.plot(loss_epoch)
        plt.title('model loss')
        plt.ylabel('loss'), plt.xlabel('epoch')
        plt.legend(['training loss'], loc = 'upper left')
        #plt.show()
        plt.savefig('loss.png')
        plt.close()
    
        plt.figure()
        plt.plot(train_acc)
        plt.plot(test_acc)
        plt.title('model acc')
        plt.ylabel('acc (%)'), plt.xlabel('epoch')
        plt.legend(['training acc', 'testing acc'], loc = 'upper left')
        #plt.show()
        plt.savefig('acc.png')
        plt.close()
    
        path = 'cifar_lenet5.pth'
        torch.save(model.state_dict(), path)

if __name__ == '__main__':
    train()