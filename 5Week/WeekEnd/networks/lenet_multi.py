import torch.nn as nn
import torch

class lenetLinear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #필요한 모듈 생성
        #까먹지말고 conv 뒤에는 bn이랑 activate 함수 넣기

        #Conv1 -> Conv1_1, 1_2, 1_3, 1_4
        #1~3 : K5,S1, P2 / 4 : K5, S1, P0
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, 
                                            kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(6),
                                nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=6, 
                                            kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(6),
                                nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=6, 
                                            kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(6),
                                nn.ReLU())
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, 
                                            kernel_size=5, stride=1, padding=0),
                                nn.BatchNorm2d(6),
                                nn.ReLU())

        self.conv1_block = nn.Sequential(
            self.conv1_1,
            self.conv1_2,
            self.conv1_3,
            self.conv1_4,
        )
        self.pool1 = nn.MaxPool2d(2,2)

        #Conv2 -> Conv2_1, 2_2, 2_3, 2_4
        #1~3 : K5,S1, P2 / 3 : K5, S1, P0
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, 
                                            kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(16),
                                nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, 
                                            kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(16),
                                nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, 
                                            kernel_size=5, stride=1, padding=0),
                                nn.BatchNorm2d(16),
                                nn.ReLU())
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv2_block = nn.Sequential(
            self.conv2_1,
            self.conv2_2,
            self.conv2_3,
        )

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
        x = self.conv1_block(x)
        x = self.pool1(x)
        x = self.conv2_block(x)
        x = self.pool2(x)
        # 중간에 fc를 태우기 전에 reshape해야하고
        _, channel, height, width = x.shape
        x = torch.reshape(x, (batch_size, channel,height,width))
        # reshape 된 feature가 다시 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
         
        return x