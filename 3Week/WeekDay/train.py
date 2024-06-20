#패키지 임포트
import torch
import torch.nn as nn

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

#모델 관련
input_size = 28*28
hidden_size = 500
output_size = 10
##데이터 관련
batch_size = 100
##학습관련
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

##데이터 불러오기##
#MNIST dataset 모듈을 불러오기
train_dataset = MNIST(root='../../data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='../../data', train=False, download=True, transform=ToTensor())

#DataLoadr 모듈 만들기
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#모델 만들기#
class mlp(nn.Module):
    def __init__(self, input_size = 28*28 , hidden_size=500, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,output_size)

    def forward(self, x): #x:데이터(이미지, batch_size x channel x height x width)
        batch_size, channel, height, width = x.shape
        batch_size, height, width = x.shape

        #4개의 nn.linear를 이용(전체 입력: 28x28, Hidden layer수: 3, 출력: 10)
        # -> fc1 -> fc2 -> fc3 -> fc4
        x = x.reshape(batch_size, height*width)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

#모델 객체를 생성(설계도 + Hparam)
model = mlp(input_size, hidden_size, output_size).to(device)


# Loss 계산하는 계산기(분류문제, classification-> cross-entropy)
critera = CrossEntropyLoss()

# Optimizer(Adam)
optim = Adam(params=model.parameters(), lr=lr)

# for loop를 돌면서 데이터 불러오기
for data, label in train_loader:
    data = data.to(device)
    label = data.to(device)

    #불러온 데이터 모델에 넣기
    output = model(data)

    #나온 출력물(output)로 loss를 계산
    loss = critera(output, label)
    #loss로 back prop 진행
    loss.backward()
    #optimizer를 이용해 최적화를 진행해야함
    optim.step()
    optim.zero_grad()

    loss += loss.item()

    #학습 중간에 평가를 진행해서
    #성능이 좋으면 저장을 진행

