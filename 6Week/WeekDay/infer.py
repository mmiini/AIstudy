# 패키지를 임포트 
import torch

from networks.mymlp import mlp
from utils.parsing import test_parse_args
from utils.utils import (get_target_infer_image, get_hparam, 
                        get_trained_weight, postprocess_image)

def main(): 
    args = test_parse_args()

    # 추론에 필요한 데이터 준비 
    image_tensor = get_target_infer_image(args)
    
    ## 학습이 완료된 최고의 모델을 준비하기 ##
    # 저장이 된 폴더를 지정받고 -> args에서 진행 
    # 거기서 필요한 내용(Hparam, Weight)을 가지고 오기 
    hparam = get_hparam(args)
    weight = get_trained_weight(args)

    # device 설정 
    args.device='cuda' if torch.cuda.is_available() else 'cpu' 

    # 설계도 + Hparam 모델 껍대기 만들기 
    model = mlp(hparam['input_size'], hparam['hidden_size'], hparam['output_size'])
    
    # 속이 빈 모델에 학습된 모델의 weight를 덮어 씌우기 
    model.load_state_dict(weight)
    model = model.to(args.device)

    # 준비된 데이터를 모델에 넣기 
    output = model(image_tensor)

    ## 결과를 분석 ## 
    # 결과를 사람이 이해할 수 있는 형태로 변환
    prob, predict = postprocess_image(output)
    print(f'해당 이미지를 보고 딥러닝 모델은 {prob:.2f}% 의 확률로 {predict}이라고 대답했다.')

    # 모델의 추론 결과를 보고 객관적인 평가 내려보기 

if __name__ == "__main__":
    main()