# 패키지 임포트 
import os 
import json 
import torch 
import torch.nn as nn 

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# hyper-parameters 설정 
## 모델 관련 
input_size=28*28
hidden_size=50
output_size=10
## 데이터 관련 
batch_size=100
## 학습 관련 
lr=0.001
epochs=10
## 폴더 세팅 
save_folder_name = 'save'

# 저장에 필요한 최상위 폴더를 만들고 
## 만약 최상위 폴더가 없으면 만든다. 
if not os.path.exists(save_folder_name): 
    os.makedirs(save_folder_name)
# 최상위 폴더 내에서 이전 학습 결과 폴더(ex. 1,2,3) 다음에 해당하는 폴더(ex. 4)를 만든다 
## 최상위 폴더 안에 있는 폴더를 쭉 보고 
prev_folder = [int(f) for f in os.listdir(save_folder_name) 
               if os.path.isdir(os.path.join(save_folder_name, f))] + [0]
## 그 폴더들 이름에서 제일 큰 수를 찾고 
prev_max = max(prev_folder)
## 그 수보다 하나 더 큰 수의 이름을 갖는 폴더를 만든다 
save_folder = os.path.join(save_folder_name, str(prev_max  + 1))
os.makedirs(save_folder)
# 그리고 설정된 hyper-parameters 저장 : json의 형태로 저장 (json)
hparam = { 
    'input_size' : input_size,
    'hidden_size' : hidden_size,
    'output_size' : output_size,
    'batch_size' : batch_size,
    'lr' : lr,
    'epochs' : epochs,
}
with open(os.path.join(save_folder, 'hparam.json'), 'w') as f: 
    json.dump(hparam, f, indent=4)

device='cuda' if torch.cuda.is_available() else 'cpu' 

## 데이터 불러오기 ## 
# MNIST dataset 모듈을 불러오기 
train_dataset = MNIST(root='../../data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='../../data', train=False, download=True, transform=ToTensor())
# dataloader 모듈을 만들기 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

## 모델 만들기 ##
class mlp(nn.Module):
    ## 모델 설계도 만들기 
    def __init__(self, input_size, hidden_size, output_size): 
        super().__init__()
        # 데이터 펼치기 -> fc1 -> fc2 -> fc3 -> fc4 -> 출력 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x): # x : 데이터 (이미지, batch_size x channel x height x width)
        batch_size, channel, height, width = x.shape

        # 4개의 nn.linear를 이용 (전체 입력 : 28*28, Hidden layer수 : 3, 출력 : 10)
        x = x.reshape(batch_size, height*width)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

# 모델 객체를 생성 (설계도 + Hparam)
model = mlp(input_size, hidden_size, output_size).to(device)

# Loss 계산하는 계산기 (분류 문제, classification -> cross-entropy)
critera = CrossEntropyLoss()

# Optimizer (Adam)
optim = Adam(params=model.parameters(), lr=lr)

def evaluate(): 
    with torch.no_grad():
        model.eval()
        ## 평가에 필요한 타겟 데이터를 가지고와야함 (MNIST : 10,000개) -> for문이 필요
        total = 0 
        correct = 0 
        for data, label in test_loader: 
            ## 모델에 데이터를 넣어주고, 결과를 출력 
            data, label = data.to(device), label.to(device)
            output = model(data)

            ## 출력된 결과가 정답이랑 얼마나 비슷한지를 확인 -> correct 수 
            _, indices = torch.max(output, dim=1)
            correct += (label == indices).sum().item()
            ## total 수도 준비 
            total += label.shape[0]
        ## acc = correct / total 
        acc = correct / total 
        model.train()

    return acc

best_acc = 0 
# 전체 데이터를 학습하는 과정을 N회 진행 
for epoch in range(epochs): 
    # for loop를 돌면서 데이터를 불러오기 
    for idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        # 불러온 데이터를 모델에 넣기 
        output = model(data)
        
        # 나온 출력물(output)로 loss를 계산 
        loss = critera(output, label)
        # Loss로 back prop 진행 
        loss.backward() 
        # optimizer를 이용해 최적화를 진행 
        optim.step()
        optim.zero_grad()
        
        if idx % 100 == 0 : 
            print('Loss의 값은 : ', loss.item())
            
            # 학습 중간에 평가를 진행해서 
            acc = evaluate()
            # 성능이 좋으면 
            if best_acc < acc : 
                best_acc = acc 
                print('Best acc : ', acc)
                # 모델의 weight 저장을 진행 
                torch.save(model.state_dict(), 
                           os.path.join(save_folder, 'best_model.ckpt'))
                # 필요시 meta data를 저장 