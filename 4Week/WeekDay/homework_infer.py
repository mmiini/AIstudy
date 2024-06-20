#패키지 임포트
import torch
import os
import json
import torch.nn as nn

from PIL import Image
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax

#device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#추론에 사용할 데이터 불러오기
example_image_path = 'mnist_example.jpeg'
image = Image.open(example_image_path)

#전처리 과정 : train.py에서 진행한 전처리와 같아야한다
# 28x28 이미지 크기 변경
image = image.resize((28,28))

#RGB 이미지 -> Gray scale 이미지 변경
image = image.convert("L")
#ToTensor()적용
image_tensor = ToTensor()(image)
image_tensor = image_tensor.unsqueeze(0).to(device)


#학습이 완료된 최고의 모델을 준비하기#
#저장이 된 폴더 지정
trained_folder_path = 'save/2'

#필요한 내용(Hparam, weight)을 가져오기
hparam_path = os.path.join(trained_folder_path, 'hparam.json')
with open(hparam_path, 'r')as f:
    hparam = json.load(f)

weight_path = os.path.join(trained_folder_path, 'best_model.ckpt')
weight = torch.load(weight_path)

# 설계도 + Hparam 모델 껍대기 만들기 
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
    

model = mlp(hparam['input_size'], hparam['hidden_size'],hparam['output_size'])

#빈모델에 학습 모델 weight 넣기
model.load_state_dict(weight)
model = model.to(device)

#준비된 데이터 모델에 집어넣기
output= model(image_tensor)

##결과를 분석하기##
#결과를 사람이 이해할 수 있는 상태로 변경하기#
probability = softmax(output, dim=1) #softmax 확률 형태로 변경
#가장 높은 값과 가장 높은 값의 예측 클래스
values, indices = torch.max(probability, dim=1)
prob = values.item()*100
predict = indices.item()

print(f'해당 이미지를 보고 딥러닝 모델은 {prob:.2f}%의 확률로 {predict}이라고 대답했다.')