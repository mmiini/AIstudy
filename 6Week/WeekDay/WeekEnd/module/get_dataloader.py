from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import Normalize


#cifar10 이미지의 평균, 표준편차 값
CIFAR_MEAN = [0.491,0.482,0.447]
CIFAR_STD = [0.247,0.244,0.262]

def get_target_dataset(args):
    #내가 목표하는 데이터셋이 무엇인지에 따라 분기 제공
    if args.dataset == 'mnist':
        #MNIST dataset 모듈을 불러오기
        from torchvision.datasets import MNIST
        train_dataset = MNIST(root='../../data', train=True, download=True, transform=ToTensor())
        test_dataset = MNIST(root='../../data', train=False, download=True, transform=ToTensor())
    elif args.dataset == 'cifar':
        #CIFAR10 dataset 모듈을 불러오기
        from torchvision.datasets import CIFAR10
        transform= Compose([Resize((args.img_size,args.img_size)),
                            ToTensor(), #cifar10이미지는 32x32 크기이므로
                            Normalize(mean = CIFAR_MEAN, std = CIFAR_STD)
                            ])

        train_dataset = CIFAR10(root='../../data', train=True, transform=transform, download=True )
        test_dataset = CIFAR10(root='../../data', train=False, transform=transform, download=True )
    else:
        raise TypeError('타겟 데이터셋이 잘못되었습니다.')
    return train_dataset, test_dataset


def get_target_loader(args):
##데이터 불러오기##
    train_dataset, test_dataset = get_target_dataset(args)

    #DataLoadr 모듈 만들기
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args. batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader