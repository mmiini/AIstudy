import argparse

def train_parse_args():
    parser = argparse.ArgumentParser()
    ##모델 관련
    parser.add_argument('--input_size', type=int, default=28*28, help='모델에 들어가는 데이터의 크기입니다.')
    
    #MLR
    parser.add_argument('--hidden_size', type=int, default=50, help='모델 hidden layer의 크기입니다.')
    parser.add_argument('--output_size', type=int, default=10, help='모델의 출력')
    parser.add_argument('--model_type', type=str, default='lenet', choices=['lenet', 'mlp', 'lenet_linear','lenet_multi', 'lenet_incept'], help='모델')
    
    #Lenet
    parser.add_argument('--mid_feature', type=int, default=2048, help='LeNet Linear 모델에서 중간 레이어 피쳐를 의미ㅏㅁ')
    #데이터 관련
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='cifar', choices=['mnist','cifar'])
    parser.add_argument('--num_classes', type=int, default=10 )
    #학습 관련
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--epochs', type=int, default=10, help='학습량(에폭)')
    #폴더 세팅
    parser.add_argument('--save_folder_name', type=str, default='save', help='저장 폴더 묶어주는 상위 폴더 이름')
    parser = parser.parse_args()
    return parser


def test_parse_args():
    parser = argparse.ArgumentParser()
    ##어떤 모델 쓸건지
    parser.add_argument('--example_image_path', type=str, default='mnist_example.jpeg')
    #어떤 데이터 쓸껀지
    parser.add_argument('--trained_folder_path', type=str, default='save/2')
    
    parser = parser.parse_args()
    return parser