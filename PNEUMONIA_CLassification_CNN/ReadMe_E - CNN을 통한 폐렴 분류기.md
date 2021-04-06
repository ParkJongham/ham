# CNN 을 통한 폐렴 분류기

## 의료 영상

오늘날 많은 질병을 진단하는데 있어서 X RAY, CT, MRI 등을 활용한다. 이렇게 촬영된 데이터 (영상, 이미지) 를 활용하여 질병을 분류할 수 있다.

## 의료 영상 자세 분류

1. Sagittal plane : 시상면. 사람을 왼쪽과 오른쪽으로 나누는 면.
2. Coronal plane : 관상면. 인체를 앞뒤로 나누는 면.
3. Transverse plane : 횡단면 (수평면). 인체를 상하로 나누는 면.

진단 시 영상을 볼 때 검사받는 사람이 정면을 보고 있는 것으로 가정한다. 즉, '나' 를 기준으로 왼쪽이 오른쪽, 오른쪽이 왼쪽이 된다.

## X-RAY 의 특성

전자기파가 몸을 통과한 결과를 시각화한 이미지로 흑백의 염망 이미지로 생성된다.

하얀색은 뼈, 근육 및 지방은 연한 회색, 공기는 검은색으로 나타난다. 즉, 밀도가 높은 부분은 흰색, 밀도가 낮은 부분일 수록 검은색으로 나타난다.


# CNN 을 활용한 폐렴 (PNEUMONIA) 분류

## 데이터셋 설명

Chest X - RAY Images : 캐글의 데이터셋.

[데이터셋 다운로드](kaggle.com/paultimothymooney/chest-xray-pneumonia)

중국 광저우의 여성 및 어린이 병원의 1~5세 소하 환자의 흉부 X - RAY 이미지로, 총 5,856 개의 이미지이다. 

## 폐렴 구별 방법

X - RAY 이미지상 다양한 양상의 음영 (폐 부위에 희마한 그림자) 증가가 관찰된다. 

쉽게 구별이 가능하지만 실제 X - RAY 를 보면 희미한 경우가 많아 파악이 어렵다.

셰균성 폐렴은 일반적으로 오른
쪽 상부 옆에서 확인 가능한 반면, 바이러스성 폐렴은 양쪽 폐에서 보다 확산된 __interstitial (조직사이에 있는)__ 패턴이 나타난다.

즉, 일정한 패턴을 가지고 있으며, 이러한 패턴을 읽어내는 알고리짐을 딥러닝으로 구현해 보고자 한다.

## 모델 구성

[모델 작성 코드](https://github.com/ParkJongham/ham/tree/master/PNEUMONIA_CLassification_CNN)

1. 필요 라이브러리 임포트
2. 필요 변수 생성 (AUTOTUNE, ROOT_PATH, BATCH_SIZE, IMAGE_SIZE, EPOCHS)
3. 데이터 가져오기
4. train, test, val dataset 개수 확인
5. train data를 이용해 train data : val data 의 비율을 8:2 로 조정
	- val data 가 너무 적어 학습에 영향을 미칠 수 있어 비율 조정
6. train data의 정상 이미지와 폐렴 이미지 개수 확인
	- 정상 이미지보다 폐렴 이미지가 3배 이상 더 많다.
	- 이러한 경우를 클래스 불균형 (imbalance) 한다.
	- 일반적으로 CNN 모델은 킅래스가 balance 할 수록 학습을 더 잘한다.
7.  tf.data 인스턴스 구성
	- tf.data 를 통해 tensorflow 학습 시 배치처리 작업을 보다 효율적으로 수행하게 도와준다.
8. 만들어진 train, val data 개수 확인
9. 라벨 이름 확인
10. 라벨을 추출하는 get_label 함수 정의
	- 라벨 이름이 폴더명으로 되어 있어 해당 경로의 마지막 폴더명 부분을 추출
 11. 데이터 셋의 이미지를 동일한 크기로 변경
	 - 여러 의료 영상, 이미지는 사이즈가 다 다를 수 있어 사이즈를 통일 시키고, GPU 메모리 효율성을 증대시키기 위해 사이즈 축소.
	 - 먼저 정의된 get_label 함수를 이용하여 라벨 추출
	 - porcess_path 함수에서 decode_img 함수를 이용, 이미지 데이터 타입을 float로 변환 후 사이즈 변경.
12. 최종 train, val data set 생성
	- num_parallel_calss 파라미터에서 Set-up 에서 초기화 한 AUTOTUNE 을 이용하여 데이터 처리 속도를 향상
	- 생성된 데이터셋 확인 (train_ds.take(1) 은 생성한 데이터셋 에서 1개만을 가져와서 확인한다는 의미)
13. test data set 정의
	- train, val data 와 동일하게 이미지 사이즈 변경 및 축소 등 전처리
14. prepare_for_training() 함수 정의
	- tf.data 파이프라인을 활용해 학습 데이터를 효율적으로 사용
	- shuffle() : 고정 크기 버퍼를 유지하되, 랜덤하게 다음 요소를 선택
	- repest() : 학습을 수행하면서 여러번 데이터셋을 호출하는데, 이를 여러번 사용할 수 있도록 자동으로 맞춰준다
	- batch() : 위에서 정의한 BATCH_SIZE 로 지정되며, 몇 개의 특징을 가지고 학습을 수행할 지 결정
	- prefetcf() : 학습데이터를 나눠 읽어 옴으로 첫 번째 데이터를 GPU를 통해 학습하는 동안 두번째 데이터를 CPU에서 준비하여 효율성 향상 시켜준다
15. 데이터 시각화를 위한 show_batch() 함수 정의
	- train data 중 첫번재 배치를 추출
	- 추출된 배치의 이미지와 라벨을 분리
	- show_batch() 함수를 통해 시각화
16. CNN 모델링
	- convolution block 함수 정의
		- 2개의 convolution layer 를 통과
		- gradient vanishing, gradient exploding 문제를 batch normalizaion 을 통해 해결
		- Max pooling 
	- dense block 함수 정의
17. Batch Normalizaion 과 dropout, 2가지 regularizaion 을 동시에 사용하는 CNN 모델 정의
	- 일부 논분에는 variance shift를 억제하기 때문에 Batch Normalization 과 dropout 를 함께 사용하용하는 것을 금기시 한다
	- 하지만 같이 사용하는 경우가 실제로 성능 향상에 도움이 되는 경우가 많기 때문에 두 regularization 을 함께 사용
18. 6번에서 확인한 데이터 imbalance 문제를 처리
	- Weight balancing 테크닉을 통해 해결
	- Weight balanceing 는 traing dataset 의 각 데이터에서 loss 계산 시 특정 클래스에 더 큰 가중치를 부여
	- keras에서 model.fit() 을 통해 파라미터로 넘기는 class_weight 에 클래스별 가중치를 세팅할 수 있다
	- `weight_for_0` 는 'Normal' 이미지에 사용할 가중치를, `weight_for_1` 는 'Pneumonia' 이미지에 사용할 가중치를 셋팅
	- 셋팅된 가중치는 전체 데이터 건수에 반비례하도록 설정
19. 모델 훈련
	- 이미지 훈련은 GPU를 사용, 따라서 GPU 선택
	- 14번에서 정의된 CNN 모델 사용
	- 정상, 폐렴 2개의 분류를 수행하므로 'binary_cross entropy' 를 loss 값으로 설정
	- optimaizer 로 'adam' 설정
	- 성과측정을 위한 metrics지표로 'accuracy', 'precision', 'recall' 을 사용
21. 모델 훈련
21. 훈련 결과 시각화
22. 훈련된 모델을 test data 를 통해 평가
