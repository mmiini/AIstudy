import argparse

# hyper-parameters 설정 
def train_parse_args():
    parser = argparse.ArgumentParser()
    ## 모델 관련 
    parser.add_argument('--input_size', type=int, default=28*28, help='모델에 들어가는 데이터의 크기입니다.')
    parser.add_argument('--hidden_size', type=int, default=50, help='모델 hidden layer의 크기입니다.')
    parser.add_argument('--output_size', type=int, default=10, help='모델의 출력')

    ##데이터 관련
    parser.add_argument('--batch_size', type=int, default=100)

    ##학습관련
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--epochs', type=int, default=10, help='학습량(에포크)')

    ##폴더 세팅
    parser.add_argument('--save_folder_name', type=str, default='save', help='저장 폴더 묶어주는 상위 폴더 이름')

    parser = parser.parse_args()
    return parser

def test_parse_args():
    parser = argparse.ArgumentParser()
    ## 모델 관련 
    parser.add_argument('--example_image_path', type=str, default='mnist_example.jpeg', help='이미지 경로')
    parser.add_argument('--trained_folder_path', type=int, default='save/2', help='학습된 weight 저장 경로')
    
    parser = parser.parse_args()
    return parser
    