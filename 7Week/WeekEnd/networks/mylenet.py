import torch.nn as nn
import torch

class lenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #필요한 모듈 생성
        #까먹지말고 conv 뒤에는 bn이랑 activate 함수 넣기
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
         
        self.pool1 = nn.MaxPool2d(2,2)

        

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2,2)

        #reshpae은 모듈이라기보다는 크기를 변경해주는 기능이기때문에 함수로 forward다시
        self.fc1 = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, num_classes),
            nn.ReLU()
        )

        pass

    def forward(self, x):
        batch_size, channel, height, width = x.shape
        #데이터 x가 들어옴
        # x의 크기는 batch, 3, 32, 32 임
        # 이미지 크기 그대로 앞쪽 모듈부터 진행됨
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        _, channel, height, width = x.shape
        # 중간에 fc를 태우기 전에 reshape해야하고
        x = torch.reshape(x, (batch_size, channel*height*width))
        # reshape 된 feature가 다시 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return