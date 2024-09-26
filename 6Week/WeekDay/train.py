# 패키지 임포트 
import os 
import torch 

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from utils.utils import evaluate
from utils.parsing import train_parse_args
from utils.utils import make_folder, save_args, get_models
from module.get_dataloader import get_target_loader

def main():
    args = train_parse_args()
    
    # 저장할 타겟 폴더의 위치를 잡아주고 
    args.save_folder = make_folder(args.save_folder_name)
    # 그 위치에 parsing한 args를 저장한다. 
    save_args(args)

    args.device='cuda' if torch.cuda.is_available() else 'cpu' 

    # 필요한 데이터 준비 
    train_loader, test_loader = get_target_loader(args)

    # 모델 만들기
    model = get_models(args).to(args.device)

    # Loss 계산하는 계산기 (분류 문제, classification -> cross-entropy)
    critera = CrossEntropyLoss()

    # Optimizer (Adam)
    optim = Adam(params=model.parameters(), lr=args.lr)

    best_acc = 0 
    # 전체 데이터를 학습하는 과정을 N회 진행 
    for epoch in range(args.epochs): 
        # for loop를 돌면서 데이터를 불러오기 
        for idx, (data, label) in enumerate(train_loader):
            data = data.to(args.device)
            label = label.to(args.device)

            # 불러온 데이터를 모델에 넣기 
            output = model(data)
            
            # 나온 출력물(output)로 loss를 계산 
            loss = critera(output, label)
            # Loss로 back prop 진행 
            loss.backward() 
            # optimizer를 이용해 최적화를 진행 
            optim.step()
            optim.zero_grad()
            
            if idx % 100 == 0 : 
                print('Loss의 값은 : ', loss.item())
                
                # 학습 중간에 평가를 진행해서 
                acc = evaluate(model, test_loader, args)
                # 성능이 좋으면 
                if best_acc < acc : 
                    best_acc = acc 
                    print('Best acc : ', acc)
                    # 모델의 weight 저장을 진행 
                    torch.save(model.state_dict(), 
                            os.path.join(args.save_folder, 'best_model.ckpt'))
                    # 필요시 meta data를 저장 

if __name__ == "__main__":
    main()