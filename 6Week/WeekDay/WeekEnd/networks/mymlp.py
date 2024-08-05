import torch.nn as nn

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