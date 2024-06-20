from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def get_target_loader(args):
##데이터 불러오기##
    #MNIST dataset 모듈을 불러오기
    train_dataset = MNIST(root='../../data', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='../../data', train=False, download=True, transform=ToTensor())

    #DataLoadr 모듈 만들기
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args. batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader