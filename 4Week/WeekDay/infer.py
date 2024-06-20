# 패키지를 임포트 
import os 
import json 
import argparse
import torch
import torch.nn as nn 

from PIL import Image
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax

def pars():
    parser = argparse.ArgumentParser()
    ##어떤 모델 쓸건지
    parser.add_argument('--example_image_path', type=str, default='mnist_example.jpeg')
    #어떤 데이터 쓸껀지
    parser.add_argument('--trained_folder_path', type=str, default='save/2')
    
    parser = parser.parse_args()
    return parser

def main():
    args = pars()
    # 추론에 사용할 데이터를 준비 
    image = Image.open(args.example_image_path)

    ## 전처리 과정 : train.py에서 진행한 전처리와 똑같아야 해 
    ## 28x28 이미지 크기 변경도 필요 
    image = image.resize((28, 28))
    ## RGB 이미지 -> Gray scale로 변경 
    image = image.convert("L")
    ## ToTensor() 이걸 적용, 
    image_tensor = ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0).to(args.device)

    ## 학습이 완료된 최고의 모델을 준비하기 ##
    # 저장이 된 폴더를 지정받고 
    # 거기서 필요한 내용(Hparam, Weight)을 가지고 오기 
    hparam_path = os.path.join(args.trained_folder_path, 'hparam.json')
    with open(hparam_path, 'r') as f : 
        hparam = json.load(f)

    weight_path = os.path.join(args.trained_folder_path, 'best_model.ckpt')
    weight = torch.load(weight_path)

    # device 설정 
    args.device='cuda' if torch.cuda.is_available() else 'cpu' 

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
        
    model = mlp(hparam['input_size'], hparam['hidden_size'], hparam['output_size'])
    # 속이 빈 모델에 학습된 모델의 weight를 덮어 씌우기 
    model.load_state_dict(weight)
    model = model.to(args.device)

    # 준비된 데이터를 모델에 넣기 
    output = model(image_tensor)

    ## 결과를 분석 ## 
    # 결과를 사람이 이해할 수 있는 형태로 변환
    probability = softmax(output, dim=1) # softmax : 확률의 형태로 변경 
    values, indices = torch.max(probability, dim=1)
    prob = values.item()*100
    predict = indices.item()

    print(f'해당 이미지를 보고 딥러닝 모델은 {prob:.2f}% 의 확률로 {predict}이라고 대답했다.')

# 모델의 추론 결과를 보고 객관적인 평가 내려보기 


if __name__ == "__main__":
    main()