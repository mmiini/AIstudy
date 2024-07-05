import os 
import json 
import torch 
from PIL import Image
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax

def make_folder(save_folder_name): 
    # 저장에 필요한 최상위 폴더를 만들고 
    ## 만약 최상위 폴더가 없으면 만든다. 
    if not os.path.exists(save_folder_name): 
        os.makedirs(save_folder_name)
    # 최상위 폴더 내에서 이전 학습 결과 폴더(ex. 1,2,3) 다음에 해당하는 폴더(ex. 4)를 만든다 
    ## 최상위 폴더 안에 있는 폴더를 쭉 보고 
    prev_folder = [int(f) for f in os.listdir(save_folder_name) 
                if os.path.isdir(os.path.join(save_folder_name, f))] + [0] 
    ## 그 폴더들 이름에서 제일 큰 수를 찾고 
    prev_max = max(prev_folder) 
    ## 그 수보다 하나 더 큰 수의 이름을 갖는 폴더를 만든다 
    save_folder = os.path.join(save_folder_name, str(prev_max  + 1)) 
    os.makedirs(save_folder) 
    return save_folder

def save_args(args):
    # 그리고 설정된 hyper-parameters 저장 : json의 형태로 저장 (json) 
    with open(os.path.join(args.save_folder, 'hparam.json'), 'w') as f: 
        json.dump(vars(args), f, indent=4)
        # json.dump(args.__dict__, f, indent=4)


def evaluate(model, test_loader, args): 
    with torch.no_grad():
        model.eval()
        ## 평가에 필요한 타겟 데이터를 가지고와야함 (MNIST : 10,000개) -> for문이 필요
        total = 0 
        correct = 0 
        for data, label in test_loader: 
            ## 모델에 데이터를 넣어주고, 결과를 출력 
            data, label = data.to(args.device), label.to(args.device)
            output = model(data)

            ## 출력된 결과가 정답이랑 얼마나 비슷한지를 확인 -> correct 수 
            _, indices = torch.max(output, dim=1)
            correct += (label == indices).sum().item()
            ## total 수도 준비 
            total += label.shape[0]
        ## acc = correct / total 
        acc = correct / total 
        model.train()

    return acc

def get_target_infer_image(args, image_size=28):
    # 추론에 사용할 데이터를 준비 
    image = Image.open(args.example_image_path)

    ## 전처리 과정 : train.py에서 진행한 전처리와 똑같아야 해 
    ## 28x28 이미지 크기 변경도 필요 
    image = image.resize((image_size, image_size))
    ## RGB 이미지 -> Gray scale로 변경 
    image = image.convert("L")
    ## ToTensor() 이걸 적용, 
    image_tensor = ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0).to(args.device)
    return image_tensor

def get_hparam(args):
    hparam_path = os.path.join(args.trained_folder_path, 'hparam.json')
    with open(hparam_path, 'r') as f : 
        hparam = json.load(f)
    return hparam

def get_trained_weight(args):
    weight_path = os.path.join(args.trained_folder_path, 'best_model.ckpt')
    weight = torch.load(weight_path)
    return weight

def postprocess_image(output):
    probability = softmax(output, dim=1) # softmax : 확률의 형태로 변경 
    values, indices = torch.max(probability, dim=1)
    prob = values.item()*100
    predict = indices.item()
    return prob, predict


def get_models(args):
    if args.model_type =='mlp':
        from networks.mymlp import mlp
        model = mlp(args.input_size, args.hidden_size, args.output_size)
    elif args.model_type == 'lenet':
        from networks.mylenet import lenet
        model = lenet(args.num_classes)
    elif args.model_type == 'lenet_linear':
        from networks.lenet_linear import lenetLinear
        model = lenetLinear(args.num_classes, args.mid_feature)
    elif args.model_type == 'lenet_multi':
        from networks.lenet_linear import lenetMulti
        model = lenetMulti(args.num_classes)

    elif args.model_type == 'lenet_incept':
        from networks.lenet_linear import lenetIncept
        model = lenetIncept(args.num_classes)
    return


