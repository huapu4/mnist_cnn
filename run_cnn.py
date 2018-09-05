import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


batch_size = 100
# data
train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data',train=False,
                               transform=transforms.ToTensor(),
                               download=True)
print('train datasets:', train_dataset.train_data.size())
print('test datasets:', test_dataset.test_data.size())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,)

# cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),   # 16*26*26
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),   # 32*24*24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 32*12*12
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),   # 64*10*10
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3),   # 128*8*8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 128*4*4
        )
        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

print('GPU works')
model = CNN().cuda()
print(model)

optimizer = optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

for epoch in range(5):
    print('epoch', epoch+1)
    train_loss = 0
    train_acc = 0
    for x, y in train_loader:
        x, y = Variable(x.cuda()), Variable(y.cuda())
        out = model(x)
        loss = loss_func(out, y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1].cuda()
        train_correct = (pred == y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('train loss:{:.4f},train accuracy:{:.4f}'.format(train_loss / (len(train_dataset)),
                                                           train_acc / (len(train_dataset))))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for x, y in test_loader:
        x, y = Variable(x.cuda()), Variable(y.cuda())
        out = model(x)
        loss = loss_func(out, y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1].cuda()
        eval_correct = (pred == y).sum()
        eval_acc += eval_correct.item()
    print('test loss:{:.4f}, test accuracy:{:.4f}'.format(eval_loss / (len(test_dataset)),
                                                          eval_acc / (len(test_dataset))))
