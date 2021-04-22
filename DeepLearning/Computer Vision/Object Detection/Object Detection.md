# Object Detection

Object Detection 은 물체의 분류 및 위치 측정 (localization) 까지 함께 수행한다.


## 딥러닝 Object Detection 용어

1. Classification : 
- 이미지 속의 object 의 종류를 분류하는 것.

2. Localization : 
- 하나의 Object 가 있을 때 해당 object 의 위치를 특정하는 것. 주로 Bounding Box 를 사용하며, left top, right bottom 좌표를 출력한다.

3. Object Detection : 
- 여러개의 object 중에서 각 각의 object 를 파악하고, 위치를 특정하며 분류하는 작업. 여러개 혹은 단일 object 만을 특정할 수 있다.

4. Object Recognition : 
- Object Detection 과 같은 의미.

5. Object Segmentation : 
- Object Detection 을 통해 특정된 물체의 형상을 따라 영역을 표시하는 것. (포토샵에서 누끼를 따는 것)

6. Image Segmentation : 
- 이미지의 영역을 분할하는 것. image segmentation 을 합쳐서 object segmentation을 수행.

7. Semantic Segmentation : 
- object segmentation 을 하고 같은 분류의 물체를 같은 색으로 구분하여 분할하는 것.

8. Instance Segmentation : 
- 같은 분류라도 다른 instance 를 구분하는 것. 같은 분류일 때 동일한 색으로 구분하여 분할하는 것이 아닌 각각의 물체를 각기 다른 색상을 구분하여 분할하는 것.


## Object Detection  의 위치 특정 방법

1. Bounding Box : 
	- 이미지 속 특정 물체의 위치를 네모 상자로 특정하는  방법으로 네모 상자를 표현하는 2가지 방법이 존재.
		- 좌측 사단을 원점으로 정의하고, left top (좌상단), right bottom (우하단) 좌표로 표현하는 방식
		- 이미지 내의 절대 좌표로 정의하지 않고 바운딩 박스의 폭과 높이로 정의하는 방식. left top 의 점에 대한 상대적인 위치로 물체의 위치를 정의

2. Object Mask : 
	- 


## Bounding Box 를 통해 인식한 결과를 평가하는 지표

1. L1, L2 Distance : 
- L1, L2 지표를 통해 거리로 결과를 평가할 수 있지만, 박스 크기에 따라 값이 커지고 작아지는 변동이 발생하며, 이는 절대적이지 못한 단점이 존재한다.

2. IoU (Intersection over Union) : 
- L1, L2 Distance 와 달리 면적의 절대적인 값에 영향을 받지 않도록하기 위한 방법. 

- 교차하는 영역을 합친 영역으로 나눈 값이다. 
	- 만약 이미지의 바운딩 박스가 2개가 있을 경우, 알고리즘에 예측한 영역 (prediction) 과 실제 정답 영역 (ground truth) 이 존재하게 된다. 이때 예측한 영역과 실제 정답인 영역의 일치하는 영역을 예측한 영역과 실제 정답 영영의 전체 영역으로 나눠준 값. 
	- 즉 교집합인 영역을 합집한인 영역으로 나눠준 값.
	- 예측 영역과 실제 정답 영역이 일치하는 경우 IoU 는 1 이 된다.


## Localization
	
일정한 크기의 입력 이미지에 어떤 물체가 있는지 확인하고 위치를 특정하는 방법.

- Classification 모델 생성 시 convolution layer 의 구성 : 
	1. 백본 네트워크 (backbone entwork)를 통해 이미지의 특성을 추출
	2. 클래스 간 분류를 위해 fully connected layer 를 	추가
	3. localization 를 위해 박스 위치를 표현할 output 노드 4개를 추가


	- localization 을 위한 Target Label  :  $$y = \begin{bmatrix} p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \end{bmatrix}$$
		- $p_c$ : 물체가 있을 확률
		- $c_1, c_2, c_3$ : 물체가 있을 때 각 각 클래스 1, 2, 3 에 속할 확률
		- $p_c$ 가 0 인 경우 : 배경에 해당하는 경우
		- $b_x, b_y$ : 바운딩 박스의 좌측 상단의 점의 위치
		- $b_w, b_h$ : 바운딩 박스의 높이와 폭
		
		* $b_x, b_y, b_w, b_h$ : 입력 이미지의 너비 w, 높이 h 로 각각 Normalize 된 상대적인 좌표와 높이 / 폭으로 표시

예 ) Ground Truth가 되는 bounding box의 좌표(position)은 x1, y1, x2, y2 이고, 입력 image의 너비(width)는 w, 높이(height)는 h로 주어질 때 object detection을 위한 target label y는?
	- Bounding box가 x1, y1, x2, y2이고 image의 크기가 w, h 일때 Target label y=[1, x1/w, y1/h, (y2-y1)/h, (x2-x1)/w] 이다.


## 슬라이딩 윈도우 (Sliding Window)

큰 이미지에 있는 여러 물체를 한꺼번에 찾는 기법.

- 슬라이딩 윈도우의 아이디어 : 
	1. 전체 이미지를 적당한 크기의 영역으로 분할
		- Window Size :  Localization network 의 입력으로 만들어주어야 하며, 이를 위해 분할한 적당한 크기의 영역
		- 동일한 윈도우 사이즈의 영역을 Sliding 시키면서 전체 이미지에 대해 수행 (컨볼루션레이어의 커널이 이동하는 것과 동일)

	2. 각 영역에 대해 Localization 을 반복적용

* 슬라이딩 윈도우 방식의 단점 :
	- 많은 갯수의 윈도우 영역에서 localization 을 수행하기 때문에 물체의 크기가 커지면 이를 계산하는데 너무 많은 계산이 필요하게 되므로 처리속도의 문제가 심각해진다.


## 컨볼루션 (Convolution)

슬라이딩 윈도우의 단점을 개선한 방법으로 convolution 의 receptive field 를 슬라이딩 윈도우 영역으로 한다. 

예를들어 14 x 14 크기의 입력에 대해 최종적으로 1 x 1 사이즈의 출력을 한면 14 x 14 크기의 receptive field 를 슬라이딩 윈도우 영역으로 한다.

하지만 병렬적으로 동시에 localization 을 수행하므로 속도 측면에서 효율적이다.


## 앵커 박스 (Anchor Box)

이미지에 2개의 사물이 겹쳐있을 경우 (예를들면 차 앞에 사람이 서 있는 경우) 한 칸에 한가지 물체를 감지하기 때문에 2개의 사물을 모두 감지하기 어렵다.

이를 해결하기위한 방법으로 앵커 박스 라는 기법을 사용하여 서로 다른 형태의 물체와 겹친 경우에 대응.

예 ) 한 이미지에서 겹쳐있는 차와 사람 2개의 클래스에 대해 물체를 감지하기 위해 한 개의 그리드 셀에 대한 결과값 벡터가 물체가 있을 확률, 2개의 클래스, 그리고 바운딩 박스 4개로 총 7개의 차원을 가지게된다.

따라서 입력값이 16x16일때, 이 그림을 2x2로 총 4칸의 그리드로 나누었다고 하면, 결과값의 형태는 7개의 채널을 가져 2x2x7이된다.

2개의 물체를 특정하는 앵커 라벨 : 
 $$y = \begin{bmatrix} p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \\ p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \end{bmatrix}$$

$p_c$ ~ $c_3$ 의 한 구간이 1개의 물체를 담당하는 앵커박스 (라벨) 가 되며, 다른 하나의 $p_c$ ~ $c_3$ 의 한 구간이 나머지 물체를 담당하는 앵커박스 (라벨) 가 된다.

또한 액커 박스 1개당 하나 당 1개의 차원으로 2개의 앵커박스가 있다면 output dimension 은 2배인 2가 된다.

즉, 앵커 박스에 대한 output dimension 은 $[Batch size, output grid, Anchor box 개수]$ 가 된다.

한 그리드 셀에서 앵커 박스에 대한 물체 할당 역시 IoU 지표를 통해 물체를 할당한다. 인식 범위 내에 물체가 있고 두 개의 앵커 박스가 있는 경우 IoU가 더 높은 앵커 박스에 물체를 할당하게 된다.


바운딩 박스와 혼동이 있을 수 있는데, 바운딩 박스는 모델이 예측한 물체의 위치이며, 모델의 출력값이다.

하지만 앵커박스는 모델이 특정해야할 물체의 형태에 대한 가정이며, 모델의 입력값이다.


## NMS (Non - Max Suppression)

한 이미지에서 여러개의 물체를 특정하기 위해 겹친 여러 개의 바운딩 박스를 하나로 줄여줄 수 있는 또다른 기법.

- NMS : 
	- 겹친 바운딩 박스들이 있을 경우 가장 확률이 높은 박스를 기준으로 기준이 되는 IoU 이상인 것들을 제거. 즉, 딥러닝 모델에서 나온 object detection 결과 중 겹친 결과를 하나로 줄여 보다 나은 결과를 얻는다.
	
	- IoU를 기준으로 없애는 이유는 어느 정도 겹치더라도 다른 물체가 있는 경우가 있을 수 있기 때문이며, 이때 Non-max suppression은 같은 class인 물체를 대상으로 적용


## Detection Architecture

1. R - CNN :

- 물체가 있을 법한 후보 영역을 뽑아내는 Region proposal 알고리즘과 후보영역을 분류하는 CNN 을 하용한 모델.
- Proposal 을 만들어내는데 Selective search 라는 비신경망 알고리즘이 사용.
- 이 후 후보 영역의 Classification 과 바운딩 박스의 regression 을 위한 신경망 사용.


2. Fast R - CNN : 

- 후보 영역 (candidate box)의 classification 과 바운딩 박스 regression 을 위한 특성을 한 번에 추출하여 사용.
- 이미지를 Sliding Window 방식으로 잘라내는 것이 아니라 해당 부분을 CNN을 거친 특성 맵(Feature Map) 에 투영해, 특성 맵을 잘라내는 점이 R - CNN 과의 차이.
- 이미지를 잘라 개별로 CNN을 연산하던 R-CNN과는 달리 한 번의 CNN을 거쳐 그 결과물을 재활용할 수 있으므로 연산 수를 줄일 수 있다.
- 잘라낸 특성맵의 영역은 다양한 모양과 크기를 가지며, 해당 영역의 클래스를 분류를 위해 fully-connected layer에 배치(batch) 입력값을 사용해야한다. 이때 영역의 모양과 크기를 맞춰야하는 문제가 발생.
- 이를 해결하기 위해 RoI (Region of Interest) pooling 이라는 방법을 제안해서 후보 영역에 해당하는 특성을 원하는 크기가 되도록 pooling하여 사용. 즉, 미리 정해둔 크기가 나올 때까지 pooling연산을 수행하는 방법
- CNN 연산을 줄이는 반면 region proposal 알고리즘이 병목이 발생한다.


3. Faster R - CNN : 

- Fast R-CNN을 더 빠르게 만들기 위해서 region proposal 과정에서 RPN(Region Proposal Network) 라고 불리는 신경망 네트워크를 사용
- 이미지 모델을 해석하기 위해 많이 사용하는 CAM(Classification Activation Map)에서는 물체를 분류하는 태스크만으로도 활성화 정도를 통해 어느 정도 물체를 특정할 수 있는데, 이와 같이 이미지에 먼저 CNN 을 적용해 특성을 뽑아내 특성 맵만을 통해 물체를 특정한다.
- 이때 특성 맵을 보고 후보 영역을 얻는 모델이 RPN 이다.
- 후보 영역을 도출한 다음의 과정은 Faster R - CNN 과 동일하다.

4. Onee - Stage Dectector : 

	1. YOLO (You Only Look Once) : 
		- 이미지를 그리드로 구분하여 슬라이딩 윈도 기법을 컨볼루션 연산으로 대체, Fully Convolutional Network 연산을 통해 그리드 셀 별로 바운딩 박스를 얻어낸 뒤 바운딩 박스들에 대해 NMS를 한 방식
		- 그리드 셀마다 클래스를 구분하기 때문에 두 개의 클래스가 한 셀에 나타날 경우 정확성이 떨어진다.
		- 반면에 매우 빠른 인식 속도를 자랑한다.

	2. SSD (Single - Shot Multiibox Detector) :
		- YOLO 의 특성이 담고 있는 정보가 동일한 크기의 넓은 영역을 커버하기 때문에 작은 물체를 잡기에 적합하지 않은 단점을 해결하고자한 모델
		- 다양한 크기의 특성 맵을 활용, 이로부터 classification 과 바운딩 박스 regression 을 수행하며 다양한 크기의 물체에 대응하는 detection 모델을 구축


## Anchor

1. Matching

- YOLO와 Faster-RCNN에서 Anchor를 기반으로 Loss를 계산하는 방식에는 두가지 Intersection over Union (IoU)를 threshold로 사용
	- Background IoU threshold
		- IoU가 0.7이상일 경우 Foreground 로 할당
	- Foreground IoU threshold
		- 0.3 이하일 경우는 배경 (Background) 으로 할당

- 0.3과 0.7 중간인 Anchor들은 불분명한 영역으로 학습에 활용하지 않으며, 이 경우에 Loss 가 학습에 반영되지 않도록 해야한다.

- 물체 영역에 들어간 Anchor box이라도 교차 (intersection) 하는 면적이 작기 때문에 IoU가 작아 매칭이 되지 않는다. 

- 다양한 물체에 Anchor Box가 걸쳐있는 경우 가장 높은 IoU를 가진 물체의 Bounding box를 기준으로 계산

- 따라서 물체의 Anchor box의 크기나 aspect ratio를 조정. 즉 Anchor Box 를 키워줘야 한다. (만약 세로로 긴 물체를 주로 탐지해야 하면 세로로 긴 Anchor box를 많이 만들고 Matching되는 box를 늘려야 학습이 잘된다.)

- Anchor를 많이 둘 수록 더 많은 Anchor가 물체를 Detection하도록 동작하므로 더 좋은 Recall을 얻을 수 있지만 적절하지 않은 Anchor 는 틀린 Detection 결과를 만들어낼 수 있고 이는 전체적인 성능을 낮출 수 있다. 또한 Anchor 가 늘어난만큼 계산 비용이 증가하게 된다.


2. Boundung box Regression

- Anchor box 에서 Bounding box 는 각 Anchor box에 대응되는 network는 $t_x, t_y, t_w, t_h$ 4 가지 output 으로 bounding box 를 regression 해서 정확한 box 를 표현한다. (YOLOv3 기준)

- Bounding box 를 예측하기 위해 bounding box의 중심점 $b_x, b_y$ 와 width, height $b_w, b_h$ 를 예측해야 한다.
	- anchor box의 정보 $c_x, c_y, p_w, p_h$ 와 연관지어 찾는 방법을 사용
	- 기존 anchor box 위 좌측상단이 $c_x, c_y$ 이고 width, height 는 $p_w, p_h$ 이다. 이를 얼마나 x 축 또는 y 축 방향으로 옮길지 그리고 크기를 얼마나 조절해야하는지를 예측하여 물체의 bounding box를 추론한다.
