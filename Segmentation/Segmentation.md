# Segmentation 


## 01. Segmentation 의 종류


이미지에서 영역을 분리하는 기법인 세그멘테이션의 기법은 시맨틱 세그맨테이션과 인스턴스 세그맨테이션, 2가지가 있다.

### 1. 시맨틱 세그멘테이션 (semantic segmentation) : 
- 이미지 내의 물체들이 모여있는 영역을 인식 (localization) 하고 이 물체들이 뭔지 분류 (classification) 하는 접근할 뿐 해당 물체가 여러개 있을 경우 각각이 뭔지 구분하진 않는 접근 방식
	- 대표적인 시멘틱 세그멘테이션 모델인 U - Net 구조를 보면 2가지 클래스를 가진 세그멘테이션 맵 (segmentation map) 이 나오는데, 이때 클래스를 문제에 따라 달리 정의하면 클래스에 따른 시맨틱 세그맨테이션 맵을 얻을 수 있다.
	- 예 ) 인물 사진 모드 : 사람의 영역, 배경 클래스
	- 예 ) 의료인공지능 : 세포 사진에서 병이 있는 영역, 정상인 영역
- 세그멘테이션을 위해 이미지의 각 픽셀에 해당하는 영역 클래스별 정보가 필요하기 때문에 출력값이 크다.


### 2. 인스턴스 세그멘테이션 (instance segmantation) : 
- 이미지 내의 물체들이 모여있는 영역을 인식하고 분류 뿐만 아니라 여러 물체가 있을 경우 각각의 개체 (instance) 가 뭔지 픽셀 단위로 정확히 구분하는 접근 방식
	- Mask R - CNN : 
		- 대표적인 인스턴스 세그멘테이션 모델
		- object detection 모델에서 각 개체를 구분하고 이후에 개체 별 시맨틱 세그멘테이션을 수행. MasK-R-CNN은 2-Stage Object Detection의 가장 대표적인 Faster-R-CNN을 계승한 것으로서, Faster-R-CNN의 아이디어인 RoI(Region-of-Interest) Pooling Layer (RoIPool) 개념을 개선하여 정확한 Segmentation에 유리하게 한 RoIAlign, 그리고 클래스별 마스크 분리라는 단순한 두가지 아이디어를 통해, 클래스별 Object Detection과 시멘틱 세그멘테이션을 사실상 하나의 Task로 엮어낸 것으로 평가받는 중요한 모델. 클래스에 따른 마스크를 예측할 때, 여러 가지 태스크를 한 모델로 학습하여 물체 검출의 성능을 향상한다.
		- RoIAlign을 통해 줄어든 특성에서 마스크를 예측하기 때문에 사용하려는 목적에 따라서 정확한 마스크를 얻으려는 경우에는 부적합할 수 있다.
			
			- RoIPool : RoIPool Layer는 다양한 RoI 영역을 Pooling을 통해 동일한 크기의 Feature map으로 추출해 내는 레이어. 이후 이 고정 사이즈의 Frature map 를 바탕으로 바운딩 박스와 object 의 클래스를 추론한다.
			
			- 해당 구성은 object detection 에서는 문제가 되지 않지만 세그멘테이션에서는 object 영역의 정확한 마스킹을 필요로하기 때문에 RoIPool 과정에서 Quantization이 필요하다.
			
			- RoI  영역의 가로 / 세로 픽셀 크기로 분할할 경우 가로 / 세로 픽셀 크기의 배수 만큼의 영역만큼만 적용하며,  분할된 영역 중 절반 이상이 RoI 영역에 덮힌 곳들로 맞추다보면 RoI 영역 밖이 포함되거나 RoI 영역의 일부가 버려지는 경우도 발생하며 시맨틱 세그멘테이션의 정보손실과 왜곡을 야기한다.
		
		- RoIAlign : Quantization하지 않고도 RoI를 처리할 고정 사이즈의 Feature map을 생성

			- RoI 영역을 pooling layer의 크기에 맞추어 등분한 후, RoIPool을 했을 때의 quantization 영역 중 가까운 것들과의 bilinear interpolation 계산을 통해 생성해야 할 Feature Map을 계산 


## 02. Segmentation Model


### 1. FCN (Fully Convolution Network)

- AlexNet, VGG - 16 등의 모델을 세그멘테이션에 맞게 변형한 모델

- 일반적으로 이미지의 특성을 추출하기 위한 네트워크의 뒷단에 Fully connected layer 를 붙여 계산한 클래스별 확률을 바탕으로 이미지 분류를 수행

- FCN 모델은 Fully connected layer 대신 CNN 을 붙여 이미지 분류를 수행
	- CNN 은 이미지 내 위치의 특성을 유지

	- 위치정보를 유지하면서 클래스 단위의 히트맵(heat map)을 얻어 세그멘테이션을 수행

	- 위치의 특성을 유지하면서 이미지 분류를 하기 위해서 마지막 CNN은 1 x 1의 커널 크기 (kernel size) 와 클래스의 개수만큼의 채널을 가지며, 클래스의 히트맵을 얻을 수 있다.
	
	- 히트맵의 크기는 CNN 과 pooling 레이어를 거치며 원본 이미지보다 작아지기 때문에 upsampling 을 통해 이미지를 키워준다.
	
	- Upsampling 의 방법 : 
		- 1. Nearest Neighbor :
			- scale 을 키운 위치에서 원본에서 가장 가까운 값을 그대로 적용하는 방법

		- 2. Bilinear Interpolation : 
			- 두 축에 대해 선형보간법을 통해 필요한 값을 메우는 방법 
			- upsampling 시 발생하는 빈 값을 선형보간법 (Linear interpolation) 을 사용하 매워주며, 축을 두 방향으로 활용한다.

		- 3. Transposed Convolution
			- Convolution Layer 는 커널 사이즈를 정의하고 입력된 Feature 를 Window 에 따라서 output 을 계산하는 반면, Transposed Convolution은 이와 반대의 연산을 수행한다.
			- 거꾸로 학습된 파라미터로 입력된 벡터를 통해 더 넓은 영역의 값을 추정한다.
	
	- Deconvolution과 Interpolation 방식을 활용한 upsampling 을 수행

		- Deconvolution : 컨볼루션 연산을 거꾸로 해준 것

		- Interpolation : 보간법으로 주어진 값들을 통해 추정해야하는 픽셀 (특성맵의 크기가 커지면서 메꿔줘야하는 중간 픽셀) 을 추정하는 방법. Linear interpolation과 Bilinear interpolation 2가지로 나뉜다.

			- Linear interpolation : 1차원 상의 두 개의 점 사이에서 거리 비에 따라 추정

			- Bilinear interpolation : 2차원으로 확장해서 4개의 점 사이에서 어떤 점의 값을 추정
	
	- Skip Architecture 를 수행 : 업샘플링을 통해 원하는 세그멘테이션 성능을 향상


### 2. U - Net 

U 자 모양의 네트워크 구조를 가지고 있으며, FCN에서 upsampling을 통해서 특성 맵을 키운 것을 입력값과 대칭으로 만든 모델로, 의학관련 세그멘테이션을 목적으로 개발되었다.

- 네트워크 구조
	
	- 좌측의 Contracting path와 우측의 Expansive path 두 가지로 구분

		- Contracting path : Convolution network 와 유사한 구조
			- 각 블록은 3x3 convolution 계층과 ReLu 를 가진다.
			- downsampling을 위해서 2x2의 커널을 2 stride로 max pooling을 수행
			- 다음 convolution의 채널 크기는 두 배씩 늘어나도록 설계

		- Expansive path : 
			- 각 블록에 2 x 2 up-convolution 이 붙어 채널이 절반씩 줄어들고 특성 맵의 크기는 증가
			- contracting block과 동일하게 3 x 3 convolution 이 두 개씩 사용

	- 두 Path에서 크기가 같은 블록의 출력과 입력은 skip connection 처럼 연결해주어 low - level의 feature 를 활용

	- 마지막에는 1x1 convolution으로 원하는 시맨틱 세그멘테이션 맵을 얻을 수 있다. (입력으로 572x572 크기인 이미지가 들어가고 출력으로 388x388의 크기에 두 가지의 클래스를 가진 세그멘테이션 맵(segmentation map)이 출력된다,)

		- 마지막 세그멘테이션 맵의 크기가 입력 이미지와 다른 문제는 세그멘테이션 맵을 원하는 크기로 조정하여(resize) 해결한다.

- FCN 모델과 U - net 은 구조적 차이 외에 얻을 수 있는 세그멘테이션 맵의 해상도에서 차이가 난다.

	- 타일 (Tile) 기법 : 서로 겹치는 구간으로 타일을 나누어 네트워크를 추론, 큰 이미지에서도 높은 해상도의 세그멘테이션 맵을 얻을 수 있도록하는 기법

또한 U - net 은 세포를 검출해내기 위한 목적으로 개발이 되었고, 이때문에 세포의 영역뿐만 아니라 경계를 예측해야 한다.

이 경계를 픽셀 단위로 라벨값을 매길 때, 데이터셋에 세포나 배경보다 세포 간 경계 면적의 면적이 작을 것이고, 클래스 간 데이터 양의 불균형을 유발한다.

이를 해결하기위해 분포를 고려한 weight map 을 학습에 사용하였다. 여기서 말하는 weight는 손실함수 (loss)에 적용되는 가중치를 의미하며 의료 영상에서 세포 내부나 배경보다는 상대적으로 면적이 작은 세포 경계를 명확하게 추론해 내는 것이 더욱 중요하기 때문에, 세포 경계의 손실에 더 많은 페널티를 부과하는 방식이다.


### 3. DeepLab 계열

- 네트워크 구조

	- U - net 의 Contracting path과 Expansive path의 역할을 하는 것이 DeepLab 계열에도 있는데 이를 인코더 (Encoder), 디코더 (Decoder) 라 한다.
		
		- 인코더 : 이미지에서 필요한 정보를 특성으로 추출하는 모듈
		- 디코더 : 인코더에서 추출된 특성을 이용해 원하는 정보를 예측하는 모듈

3x3 convolution 을 사용했던 U-Net 과 달리 DeepLabV3+ 는 Atrous Convolution 을 사용하고 있으며, Atrous Convolution 을 여러 크기에 다양하게 적용한 것이 ASPP (Atrous Spatial Pyramid Pooling) 이다. DeepLab V3+ 는 ASPP 가 있는 블록을 통해 특성을 추출하고 디코더에서 Upsampling 을 통해 세그멘테이션 마스크를 얻는다.

- Atrous Convolution : 

	- 띄엄 띄엄 보는 컨볼루션으로 더 넓은 영역을 보기위해 커널이 일정 간격으로 떨어져 있다. 이를 통해 레이어가 깊지 않아도 넓은 영역 정보를 커버한다.

- Spatial Pyramid Pooling : 

	- 여러 가지 스케일로 convolution 과 pooling 을 하고 나온 다양한 특성을 연결 (concatenate) 한다. 때문에 멀티스케일로 특성을 추출하는데 있어서 병렬로 수행하는 효과를 얻을 수 있다.
	
	- 입력 이미지의 크기와 관계없이 동일한 구조를 활용할 수 있는 장점이 존재하기 때문에 다양한 크기와 비율을 가진 RoI 영역에 적용할 수 있다.
	
	- 컨볼루션을 Atrous Convolution으로 바꾸어 적용한 것은 Atrous Spatial Pyramid Pooling이라고 한다.


## 세그멘테이션 모델의 평가 지표

일반적으로 세그멘테이션의 결과값은 이미지의 크기에 맞는 세그멘테이션 맵 크기와 시맨틱 클래스의 수에 맞는 채널 크기를 가진다. 이때 각 채널의 max probability 에 따라 해당 위치의 클래스가 결정된다.

이 외에 세그멘테이션 모델의 평가지표로는 픽셀별 정확도 (Pixel Accuracy) 및 마스크 IoU (Mask Intersection - over - Union) 이 있다.

1. 픽셀별 정확도 (Pixel Accuracy) 

- 이미지 분류 문제와 동일하게 픽셀별 분류 정확도를 세그멘테이션 모델을 평가하는 기준으로 삼는 것.

- 예측 결과 맵 (prediction map) 을 클래스 별로 평가하는 경우에는 이진 분류 문제 (binary classification) 로 생각해 픽셀 및 채널 별로 평가한다. 픽셀 별 이미지 분류 문제로 평가하는 경우에는 픽셀 별로 정답 클래스를 맞추었는지 여부, 즉 True/False를 구분

	- $Accuracy = \frac{TP + TN}{FP + FN + TP + TN} = 1 - Error \\ False Positive Rate = \frac{FP}{N} \\ True Positive Rate (Recall) = \frac{TP}{P} \\ Precision = \frac{TP}{TP + FP}$


2. 마스크 IoU (Mask Intersection - over - Union) 

- 물체 검출 모델을 평가할 때는 정답 라벨 (ground truth) 와 예측 결과 바운딩 박스 (prediction bounding box) 사이의 IoU (intersection over union) 를 사용

- 마스크도 일종의 영역임을 생각했을 때 세그멘테이션 문제에서는 정답인 영역과 예측한 영역의 IoU 를 계산

```
# sample for mask iou 
intersection = np.logical_and(target, prediction) 
union = np.logical_or(target, prediction) 
iou_score = np.sum(intersection) / np.sum(union)
```

마스크 IoU를 클래스 별로 계산하면 한 이미지에서 여러 클래스에 대한 IoU 점수를 얻을 수 있으며 이를 평균하여 전체 시맨틱 세그멘테이션의 성능을 평가한다.
