# 임포트할 라이브러리 불러오기
import torch
import torch.nn as nn

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

#Hparm 설정하기
input_size = 28*28
hidden_size = 500
output_size = 10
batch_size = 100
lr= 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#데이터셋 만들기(MNIST)
train_dataset = MNIST(root='../../data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='../../data', train=False, download=True, transform=ToTensor())

#데이터로더 만들기
train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_worker=0)
test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#모델(MLP) 만들기
class mlp(nn.Module):
    #모델 설계도 만들기(fc1~fc4)
    def __init__(self, input_size=28*28, hidden_size=500, output_size=10):
        super().__init___()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,output_size)
    #실제 데이터가 흘러가는 줄기 만들기
    def forward(self,x):
        batch_size, channel, height, width = x.shape
        batch_size, height, width = x.shape

        x=x.reshape(batch_size, height*width)
        x= self.fc1(x)
        x= self.fc2(x)
        x= self.fc3(x)
        x= self.fc4(x)
        return x

#모델 객체를 생성(설계도 + Hparam)
model = mlp(input_size, hidden_size, output_size).to(device)

#loss 계산하는 계산기(분류)
critera = CrossEntropyLoss()

#옵티마이저(Adam)
optim = Adam(params=model.parameters(),lr=lr)


#for 루프로 train set 돌리기
for data, label in train_loader:
    #data, label device로 casting
    data = data.to(device)
    label = label.to(device)

    #불러온 데이터 model에 넣기
    output= model(data)

    #output으로 loss값 계산 
    loss = critera(output, label)
    #loss back prop 진행
    loss.backward()
    #옵티마이저 최적화하기
    optim.step()
    optim.zero_grad()

    loss += loss.item()



