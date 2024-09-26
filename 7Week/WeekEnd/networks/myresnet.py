import torch.nn as nn
import torch

class InputPart(nn.Moudle):
    def __init__(self):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                    kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.module2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        pass

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        return x

class OutputPart(nn.Module): 
    def __init__(self, linear_input=512, linear_output=10): 
        super().__init__()
        self.module1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.module2 = nn.Linear(linear_input, linear_output)

    def forward(self, x):
        x = self.module1(x)
        x = torch.squeeze(x) 
        x = self.module2(x)
        return x 

class Block(nn.Module): 
    def __init__(self, in_channels, out_channels, size_matching=False):
        super().__init__()
        self.size_matching = size_matching
        stride = 1 
        if self.size_matching: 
            self.size_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            stride = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        ) 
        self.relu = nn.ReLU()
    def forward(self, x):
            _x = x.clone()
            
            x = self.conv1(x)
            x = self.conv2(x)
            
            if self.size_matching: 
                _x = self.size_conv(_x)
        
            x = x + _x
            x = self.relu(x)
            return x 
    
class BottleNeck(nn.Module): 
    def __init__(self, in_channels, out_channels, size_matching=False): 
        super().__init__()
        middle_channels = out_channels//4 

        # 만약 특정 Layer이고, 특정 BN인 경우에 
        # width와 height의 크기를 /2으로 줄여야 하는데, 
        # 그 조건은 외부에서 알려주고, 그 값을 바탕으로 stride =2로 변경 
        stride = 1 
        if size_matching : 
            stride = 2 

        # 모든 Layer의 첫번째 BN은 입력 채널과 출력 채널이 다르다. 
        # 따라서 이 경우에는 channel 방향으로 사이즈 매칭을 해줘야한다. 
        self.need_channel_matching = False 
        if in_channels != out_channels : 
            self.need_channel_matching = True 
            self.size_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride, 
                        padding=0),
                nn.BatchNorm2d(out_channels)
            ) 

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=middle_channels,
                      kernel_size=1,
                      stride=1, 
                      padding=0),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=middle_channels, 
                      out_channels=middle_channels,
                      kernel_size=3,
                      stride=stride, 
                      padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        ) 
        self.conv3= nn.Sequential(
            nn.Conv2d(in_channels=middle_channels, 
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1, 
                      padding=0),
            nn.BatchNorm2d(out_channels),
        ) 
        self.relu = nn.ReLU()

    def forward(self, x):
            _x = x.clone()
            
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)


            #만약에 채널 방향으로 사이즈 매칭이 필요하다면
            #_x의 채널 크기를 변경해줘야한다.
            if self.need_channel_matching:
                _x=self.size_conv(x)

            x = x + _x
            x = self.relu(x)
            return x 

class MiddlePart(nn.Module): 
    def __init__(self, target, channels=[64, 64, 128, 256, 512], layer_nums=[2, 2, 2, 2]): 
        super().__init__()
        self.target = target 
        
        # layer1에 대해서... Basic Block이 몇 개가 사용되는지 & 그 안에 구체적인 spec은 어떻게 되는지 적어주면 
        # channel과 layer_num은 resnet 종류에따라 설정이 가능 -> 외부에서 해당 값을 넣어주고 그 값을 이용 
        self.layer1 = self.make_layer(channels[0],  channels[1], layer_nums[0], False) # Block(64)  x 2
        self.layer2 = self.make_layer(channels[1],  channels[2], layer_nums[1], True) # Block(128) x 2
        self.layer3 = self.make_layer(channels[2],  channels[3], layer_nums[2], True) # Block(256) x 2
        self.layer4 = self.make_layer(channels[3],  channels[4], layer_nums[3], True) # Block(512) x 2
        
    def make_layer(self, in_channels, out_channels, _num, size_matching):
        # _num 횟수만큼 channel 스팩을 갖고 있는 block을 만들건데 
        # 그 과정은 동일한 과정이므로 for문을 사용할 수 있겠죠?? 
        # Block일지 BottleNeck일지는 이 Class 외부에서 결정해줘야하는 상황이다. 따라서 외부에서 target을 받고 그걸 통해 객체화를 진행한다. 
        layer = [self.target(in_channels=in_channels, out_channels=out_channels, size_matching=size_matching)] 
        # for 횟수 in _num: 
        for idx, _ in enumerate(range(_num-1)): 
            # layer.append(Block(우리가 넣어준 spec을 입력으로 하는))
            layer.append(self.target(in_channels=out_channels, out_channels=out_channels))
        layer = nn.Sequential(*layer)
        return layer
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)     
        x = self.layer3(x)     
        x = self.layer4(x)     
        return x 
    
class resnet(nn.Module): 
    def __init__(self, resnet_type, num_classes): 
        super().__init__()

        target, channels, layer_nums, linear_input = self.get_infos(resnet_type)
        
        self.input_part = InputPart()
        self.middel_part = MiddlePart(target, channels, layer_nums)
        self.output_part = OutputPart(linear_input=linear_input, 
                                      linear_output=num_classes) 
        
    def get_infos(self, resnet_type): 
        if resnet_type == '18': 
            layer_nums = [2,2,2,2]
            channels=[64, 64, 128, 256, 512]
            target=Block
            linear_input = 512
        elif resnet_type == '34': 
            layer_nums = [3,4,6,3]
            channels=[64, 64, 128, 256, 512]
            target=Block
            linear_input = 512
        elif resnet_type == '50': 
            layer_nums = [3,4,6,3]
            channels=[64, 256, 512, 1024, 2048]
            target=BottleNeck
            linear_input = 2048
        elif resnet_type == '101': 
            layer_nums = [3,4,23,3]
            channels=[64, 256, 512, 1024, 2048]
            target=BottleNeck
            linear_input = 2048
        elif resnet_type == '152': 
            layer_nums = [3,8,36,3]
            channels=[64, 256, 512, 1024, 2048]
            target=BottleNeck
            linear_input = 2048
        return target, channels, layer_nums, linear_input
        
    def forward(self, x): 
        x = self.input_part(x)
        x = self.middel_part(x)
        x = self.output_part(x)
        return x 