import torch.nn as nn
import torch

class vgg_block(nn.Module):
    def __init__(self, in_channel, out_channel, num_conv, has_one_filter=False):
        super().__init__()
        
        conv_list = []
        #for iter in 사용할_conv의 수:
        for idx, _ in enumerate(range(num_conv)):
            #첫번째 conv의 경우에는 in_channel을 그대로 사용함
            #그런데, 2번째 이후의 conv는 in_channel을 이전의 conv 출력 값을 그대로 사용하라
            # in_channel = in_channel if idx == 0 else out_channel 
            if idx == 0:
                in_channel = in_channel
            else :
                in_channel = out_channel

            #has_one_filter == True 의 경우 &&
            #가장 마지막 conv에만 영향을 미칠 수 있을 것
            #이때, filter사이즈를 3 -> 1로 변경
            #동시에 padding도 사이즈를 변경해야함 1 -> 0으로 변경
            one_flag = True if has_one_filter and idx == (num_conv -1 )\
                            else False
            kernel_size = 1 if one_flag else 3
            padding = 0 if one_flag else 1
            
            #conv = nn.Sequntial(~~~)
            conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        #conv_list.append(conv)
        conv_list.append(conv)
        #conv_list = nn.ModuleList(conv_list)
        self.conv_list = nn.ModuleList(conv_list)


        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        for module in self.conv_list:
            x = module(x)

        x = self.pool(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, num_classes, feature_size=4096, in_feature=25088):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_feature,feature_size),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(feature_size,feature_size),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



class vgg_A (nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()

        self.block1 = vgg_block(in_channel=3, out_channel=64,  num=1)# filter 몇짜리, channel, 3x3 conv 몇개
        self.block2 = vgg_block(in_channel=64, out_channel=128, num=1)
        self.block3 = vgg_block(in_channel=128, out_channel=256, num=2)
        self.block4 = vgg_block(in_channel=256, out_channel=512, num=2)
        self.block5 = vgg_block(in_channel=512, out_channel=512, num=2)
        if image_size == 32:
            in_feature = 512
        else :
            in_feature = 25088
        self.classifier = Classifier(num_classes, in_feature=in_feature)

        pass

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        batch_size, channel, height, width = x.shape
        #reshape해줘야함
        x = torch.reshape(batch_size,(channel * height * width))

        x = self.classifier(x)
        
        return
    



class vgg_B(vgg_A):
    def __init__(self, num_classes,image_size):
        super().__init__( num_classes,image_size)
        self.block1 = vgg_block(in_channel=3, out_channel=64,  num=2)
        self.block2 = vgg_block(in_channel=3, out_channel=64,  num=2)

class vgg_C(vgg_B):
    def __init__(self, num_classes,image_size):
        super().__init__( num_classes,image_size)
        self.block3 = vgg_block(in_channel=128, out_channel=256,  num=3, has_one_filter=True)
        self.block4 = vgg_block(in_channel=256, out_channel=512,  num=3, has_one_filter=True)
        self.block5 = vgg_block(in_channel=512, out_channel=512,  num=3, has_one_filter=True)

class vgg_D(vgg_C):
    def __init__(self, num_classes,image_size):
        super().__init__( num_classes,image_size)
        self.block3 = vgg_block(in_channel=128, out_channel=256,  num=3)
        self.block4 = vgg_block(in_channel=256, out_channel=512,  num=3)
        self.block5 = vgg_block(in_channel=512, out_channel=512,  num=3)

class vgg_E(vgg_D):
    def __init__(self, num_classes,image_size):
        super().__init__( num_classes,image_size)
        self.block3 = vgg_block(in_channel=128, out_channel=256,  num=4)
        self.block4 = vgg_block(in_channel=256, out_channel=512,  num=4)
        self.block5 = vgg_block(in_channel=512, out_channel=512,  num=4)

class vgg(nn.Module):
    def __init__(self, vgg_type, num_classes, image_size):
        super().__init__()

        if vgg_type == 'A':
            self.model = vgg_A(num_classes, image_size)

        if vgg_type == 'B':
            self.model = vgg_B(num_classes,image_size)

        if vgg_type == 'C':
            self.model = vgg_C(num_classes,image_size)

        if vgg_type == 'D':
            self.model = vgg_D(num_classes,image_size)

        if vgg_type == 'E':
            self.model = vgg_E(num_classes,image_size)

    def forward(self, x):
        x= self.model(x)
        return x
