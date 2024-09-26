import torch 
import torch.nn as nn 

class lenetIncept(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()
        
        # 필요한 모듈 생성 
        # 까먹지말고 conv 뒤에는 bn이랑 activate 함수 넣기! 

        # 1, 3, 5 커널 사이즈를 사용하는 별개의 conv2d 모듈을 만들어야하고 
        # 각 모듈의 결과를 통합하는 concat을 진행해야 함 -> forward에서 진행 
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, 
                      kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(6), 
            nn.ReLU() 
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, 
                      kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(6), 
            nn.ReLU() 
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, 
                      kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(6), 
            nn.ReLU() 
        )

        # conv1은 다 똑같은데, 입력으로 들어오는 tensor의 channel 크기가 18인 점을 유의 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=6, 
                      kernel_size=5, stride=1, padding=0), 
            nn.BatchNorm2d(6), 
            nn.ReLU() 
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0), 
            nn.BatchNorm2d(16), 
            nn.ReLU() 
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        # reshape은 모듈이라기보다는 크기를 변경해주는 기능이기때문에 함수로 forward에서 다루면됨 
        self.fc1 = nn.Sequential(
            nn.Linear(400, 120), 
            nn.ReLU() 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84), 
            nn.ReLU() 
        )
        self.fc3 = nn.Linear(84, num_classes) 


    def forward(self, x): 
        batch_size, channel, height, width = x.shape
        # 데이터가 x라고 들어온다
        # x의 크기는 batch, 3, 32, 32 임 
        # 이미지 크기 그대로 앞쪽 모듈부터 진행됨 

        # 1, 3, 5 필터 크기를 갖는 각각의 모듈을 통과 
        x1 = self.conv1_1(x) 
        x2 = self.conv1_2(x) 
        x3 = self.conv1_3(x) 
        # 각 conv2d 모듈의 결과를 concat 진행 
        x = torch.cat([x1, x2, x3], dim=1) # batch_size, 18, 32, 32

        # concat의 결과를 conv1에 넣어줌 
        x = self.conv1(x) 
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # 중간에 fc를 태우기 전에 reshape 해야하고 
        _, channel, height, width = x.shape
        x = torch.reshape(x, (batch_size, channel * height * width)) # 400 이 됨 
        # reshape 된 feature가 다시 모듈에 들어가서 출력 됨
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x 