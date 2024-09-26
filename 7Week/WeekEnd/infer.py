# 패키지를 임포트 
import torch
import argparse

from networks.mymlp import mlp
from utils.parsing import test_parse_args
from utils.utils import (get_target_infer_image, get_hparam, 
                        get_trained_weight, postprocess_image, get_models)

def main(): 
    args = test_parse_args()
    
    ## 학습이 완료된 최고의 모델을 준비하기 ##
    # 저장이 된 폴더를 지정받고 -> args에서 진행 
    # 거기서 필요한 내용(Hparam, Weight)을 가지고 오기 
    hparam = get_hparam(args)

    # args는 Namespace 객체이고 hparam은 json 이므로 
    # args를 json으로 바꾸고 
    json_args = vars(args)
    # args랑 hparam을 합쳐서 
    json_args.update(hparam)
    # args라는 새로운 NameSpace객체를 만든다. 
    args = argparse.Namespace(**json_args)

    # device 설정 
    args.device='cuda' if torch.cuda.is_available() else 'cpu' 
    
    weight = get_trained_weight(args)

    # 추론에 필요한 데이터 준비 
    image_tensor = get_target_infer_image(args)

    # 설계도 + Hparam 모델 껍대기 만들기 
    model = get_models(args)
    
    # 속이 빈 모델에 학습된 모델의 weight를 덮어 씌우기 
    model.load_state_dict(weight)
    model = model.to(args.device)
    
    # 추론 과정은 학습 과정과 다르므로 eval을 설정 
    model.eval() 

    # 준비된 데이터를 모델에 넣기 
    output = model(image_tensor)

    ## 결과를 분석 ## 
    # 결과를 사람이 이해할 수 있는 형태로 변환
    prob, predict = postprocess_image(output)
    print(f'해당 이미지를 보고 딥러닝 모델은 {prob:.2f}% 의 확률로 {predict}이라고 대답했다.')

    # 모델의 추론 결과를 보고 객관적인 평가 내려보기 

if __name__ == "__main__":
    main()