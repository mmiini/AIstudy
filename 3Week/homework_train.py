# 패키지 임포트 
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# haper-parameters 설정 
input_size=28*28
hidden_layer=500
output_size=10

batch_size=100

lr=0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## 데이터 불러오기 ## 
# MNIST dataset 
train_dataset = MNIST(root='../../data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='../../data', train=False, download=True, transform=ToTensor())
# dataloader 
train_dataLoader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
test_dataLoader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=0)


## 모델 만들기 ##
class mlp(nn.Module):
    ## 모델 설계도 만들기 (fc1 ~ fc4)
    def __init__(self, input_size=28*28, hidden_layer=500 , output_size=10):
        super().__init___()
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, hidden_layer)
        self.fc4 = nn.Linear(hidden_layer, output_size)
    ## 실제 데이터가 흘러가는 줄기 만들기 
    def forward(self, x):
        # 데이터 펼치기 -> fc1 -> fc2 -> fc3 -> fc4 -> 출력 
        batch_size, channel, height, width = x.shape
        x=x.reshape(batch_size, height*width)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)

# 모델 객체를 생성 (설계도 + Hparam)
model = mlp(input_size, hidden_layer, output_size).to(device)
# Loss 계산하는 계산기 (분류 문제)
critera = CrossEntropyLoss()
# Optimizer (Adam)
optim= Adam(params=model.parameters(), lr=lr)


# for loop를 돌면서 데이터를 불러오기 
for data, label in train_dataLoader:
    # device로 casting 
    data = data.to(device)
    label = label.to(device)
    # 불러온 데이터를 모델에 넣기 
    output = model(data)
    # 나온 출력물(output)로 loss를 계산 
    loss = critera(output, label)
    # Loss로 back prop 진행 
    loss.backward()
    # optimizer를 이용해 최적화를
    optim.step()
    optim.zero_grad()

