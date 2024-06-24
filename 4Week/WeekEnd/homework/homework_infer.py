#패키지 임포트
import torch
import os
import json
import torch.nn as nn
import argparse

from PIL import Image
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax
from utils.parsing import test_parse_args
from networks.mymlp import mlp
from utils.utils import (get_target_infer_image, get_hparam, 
                        get_trained_weight, postprocess_image)


def main():
    args = test_parse_args()
    #device 설정
    image_tensor = get_target_infer_image(args)
    #학습이 완료된 최고의 모델을 준비하기#
    #저장이 된 폴더 지정
    #필요한 내용(Hparam, weight)을 가져오기
    hparam = get_hparam(args)
    weight = get_trained_weight(args)

    # device 설정 
    args.device='cuda' if torch.cuda.is_available() else 'cpu' 

    model = mlp(hparam['input_size'], hparam['hidden_size'],hparam['output_size'])

    #빈모델에 학습 모델 weight 넣기
    model.load_state_dict(weight)
    model = model.to(args.device)

    #준비된 데이터 모델에 집어넣기
    output= model(image_tensor)

    ## 결과를 분석 ## 
    # 결과를 사람이 이해할 수 있는 형태로 변환
    prob, predict = postprocess_image(output)
    print(f'해당 이미지를 보고 딥러닝 모델은 {prob:.2f}% 의 확률로 {predict}이라고 대답했다.')

    # 모델의 추론 결과를 보고 객관적인 평가 내려보기 

if __name__ == "__main__":
    main()