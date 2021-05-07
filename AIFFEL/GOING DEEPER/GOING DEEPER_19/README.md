# 19. 사람의 몸짓을 읽어보자
<br></br>
## Body Language, 몸으로 하는 대화

Human Pose Estimation 는 2D 와 3D 로 구분된다.
2D 는 2D 이미지에서 (x, y) 의 2차원 좌표를 찾아내며, 3D 는 2D 이미지에서 (x, y, z) 3차원 좌표를 찾아내는 기술이다.

하지만 어떻게 2차원에서 3차원 이미지를 복원할 수 있을까? 이는 굉장히 어려운 과정이다.

카메라 행렬에서 (x, y, z) world 좌표가 이미지 (u, v) 좌표계로 표현될 떄 거리축 (z 축) 정보가 소실되기 때문이다.

하지만 사람의 몸은 3D 환경에서 제약이 있기 때문에 어느정도 문제를 해결할 수 있다.

+ 참고 : 
	-   [영상 Geometry #1 좌표계](https://darkpgmr.tistory.com/77)
	-   [영상 Geometry #7 Epipolar Geometry](https://darkpgmr.tistory.com/83?category=460965)
<br></br>

## Pose 는 Face Landmark 랑 비슷해요

Human Pose 는 Face Landmark 와 서로 입력과 출력의 개수만 다를 뿐 매우 흡사하다. 
하지만 Face Landmark 는 물리적으로 거의 고정되어 있는 반면 Human Pose 는 팔, 다리가 상대적으로 넓은 범위와 자유도 (가동 범위가 넓다) 를 갖는다는 것을 고려해야하기 때문에 훨씬 높은 난이도를 자랑한다.

이렇게 자유도가 높은 것은 데이터의 분포를 특정하기가 어렵다는 것인데, 이는 더 많은 양의 데이터를 필요로하고, 복잡한 모델을 사용해야한다는 것을 의미한다.

<br></br>
### 우리에게 맞는 방법은 뭘까?

Human Pose 의 접근법은 Top - Down 방법과 Bottom - Up 방법 2가지로 나눌 수 있다.

1. Top - Down 방법
	- 모든 사람의 정확한 Keypoint 를 찾기위해 Object Detection 을 사용한다.
	- Crop 한 이미지 내에서 Keypoint 를 찾아내는 방법으로 표현한다
	- Detector 가 선행되어야하며 모든 사람마다 알고리즘을 적용해야하므로 등장 인물이 많아지면 속도가 느리다.

2. Bottom - Up 방법
	- Detector 가 없으며, Keypoint 를 먼저 검출한다. (특정 부위에 해당하는 모든 점들을 검출한다.)
	- 한 사람에게 해당하는 Keypoint 를 클러스터링 한다.
	- Detector 가 없기 때문에 다수의 사람이 등장 하더라도 속도의 저하가 크지 않다.
	- Top - Down 방식에 비해 Keypoint 검출 범위가 넓어 성능이 비교적 떨어진다.

<br></br>
이렇게 필요 정확도 및 등장 인물이 얼마나 되는지 등 목적에 따라 필요한 알고리즘은 달라진다.

핸드폰 어플을 통한 서비스에는 사람이 많이 등장하지 않기 때문에 Top - Down 방식을 사용해도 큰 속도저하가 일어나지 않을 수 있다.
<br></br>
## Human Keypoint Detection (01)

Top - Down 의 방식에 대해 자세히 알아보자.

<br></br>
### 자유도가 높은 사람의 동작

Face Landmark 와 Human Pose Estimation는 비슷하지만, 팔, 다리, 손목 등의 여러 관절의 Joint Keypoinr 정보는 얼굴의 Keypoint 보다 훨씬 자유롭다. 즉, 위치의 변화가 다양한다.

![](https://github.com/Team-Neighborhood/Kalman-Filter-Image/raw/master/result/KF_result.gif)
<br></br>
위 자료에서 볼 수 있듯이 손이 얼굴을 가리는 행위, 모든 keypoint 가 영상에 담기지 않는 등 invisible , occlusions, clothing, lighting change 가 face landmark 에 비해 더 어려운 환경을 가진다.

먼저 Human Keypoint Estimation 의 가장 기본적인 아이디어는 "인체는 변형 가능한 부분으로 나누어져 있으며, 각 부분은 연결성을 가지고 있다." 는 것이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/05_9oDmNOY.max-800x600.png)
<br></br>
위 그림에서 손은 팔과 연결되어 있고, 팔은 몸과 연결되어 있다. 이러한 제약 조건을 좌측의 스프링으로 표현한 것이다.

이는 3D 환경 측면에서는 정말 좋은 접근 방법이다. 하지만 2D 이미지 데이터를 다뤄야하는 입장에서는 별로 좋지 않다.

3D 환경에서 손이 다리 옆에 있을 확률이 팔 옆에 있을 확률 보다 작을 것이다. 하지만 2D 환경에서는 촬영 각도나 다른 조건에 따라 충분히 일어날 수 있는 일이기 때문이다.

이러한 문제를 해결하기위한 방법으로 Deformable Part Models 방법에서는 각 부분들의 Complex Joint Relationship 의 Mixture Model 로 Keypoint 를 표현하는 방법을 이용했다. 하지만 성능은 기대에 미치지 못했다.

+ 참고 : [논문 : Articulated human detection with flexible mixtures-of-parts](https://www.cs.cmu.edu/~deva/papers/pose_pami.pdf)
<br></br>

### DeepPose

Human Pose Estimation 의 한계는 위 논문에서 언급된 바와 같이 Graphical Tree Model 은 같은 이미지에 두 번 연산을 하는 등 연산 효율이 떨어지며, 부족한 성능이다.

하지만 딥러닝이 적용되면서 Pose Estimation 에 CNN 방법을 적용하게 되었다.

Toshev and Szegedy 는 처음으로 딥러닝 기반 keypoint localization 모델을 제안했다.
<br></br>
+ 참고 : -   [DeepPose: Human Pose Estimation via Deep Neural Networks](https://arxiv.org/pdf/1312.4659.pdf)
<br></br>
이렇게 기술적으로 해결이 어려웠던 부분을 딥러닝을 적용하면서 딥러닝 기반의 추론 방법이 좋은 해결책이 될 수 있음을 보였다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/07_GOoFtrb.png)
<br></br>
초기의 pose estimation 모델은 x,y 좌표를 직접적으로 예측하는 position regression 문제로 인식하여 human detection 을 통한 crop 된 사람 이미지를 이용해서 딥러닝 모델에 입력하고 (x,y) 좌표를 출력하도록 만들었다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/08_iBnIDWo.png)
<br></br>
하지만 DeepPose 가 혁신적인 시도였던 것에 반해 성능이 매우 좋지는 않았다.

위 그림을 보면 전반적인 성능의 향상을 가져왔지만 기존 방법에 의해 압도적인 성능은 아니다.
<br></br>

### Efficient Object Localization Using Convolution Network

DeepPose 모델에서 딥러닝을 적용했음에도 성능이 비약적으로 높지 않았던 이유를 Tompson 이 제안한 Efficient object localization 논문에서 말하고 있다고 할 수 있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/09_NSfxHyY.max-800x600.png)
<br></br>

Efficient object localization 에서 제안한 모델 역시 DeepPose 모델에 비해 깊은 구조를 가지고 있지만 Keypoint 를 직접 예측하지 않고 Keypoint 가 존재할 확률 분포를 학습하게 한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/10_SBHBawM.max-800x600.png)
<br></br>

Human Pose 도 사람이 라벨링을 할 수 밖에 없다. 과연 사람이 항상 같은 위치에 점을 찍을 수 있을까?
<br></br>
![](https://github.com/Team-Neighborhood/Kalman-Filter-Image/raw/master/result/KF_result.gif)
<br></br>
위 자료를 다시 보면 아래 Orig Measured 는 점들이 굉장히 떨리고 있다는 것을 알 수 있다.

이는 사람이 라벨링을 하는 과정에서 같은 위치라고 생각했지만 조금씩의 오차가 발생했기 때문이다. 즉, 특정 위치에Keypoint 를 선택하지만 특정 위치를 중심으로 "어떤 분포" 의 에러가 더해져서 저장 되는 것이기 때문이다.

일반적으로 자연상태에서 일어나는 확률은 가우시안 분포 (정규분포) 일 가능성이 높다. Tompson 은 이런 점에 착안하여 label 을 (x,y) 좌표에서 (x,y) 를 중심으로 하는 heatmap 으로 변환하여 이를 학습하게 하였다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/11_bVXlDCF.png)
<br></br>
이를 통해 2배가 넘는 성능의 비약적 향상됨을 알 수 있다.

MPII 데이터는 2014년에 나온 데이터이며, 기존 FLIC 데이터가 머리, 어깨, 팔꿈치, 손목 수준의 적은 개수의 keypoint를 가지고 있었지만 MPII는 몸의 각 관절부위 16개의 keypoint를 갖는다. 이는 기존 논문 (Gkioxari, Sapp) 들이 일부 데이터가 없는 이유이다.
<br></br>
+ 참고 : [논문 : Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-9-L-Casecaded.max-800x600.png)
<br></br>
Tompson 이 제안한 모델은 Coarse model 과 fine model 로 나누어지는데 coarse model 에서 32 x 32 heatmap 을 대략적으로 추출한 후 multi resolution 입력을 coarse heatmap 기준으로 crop 한 뒤 fine model 에서 refinement 를 수행한다.

또한 coarse model 과 fine model 이 같은 모델이며 weight 를 공유한다. 목적이 같기 때문에 빠른 학습이 가능하고 메모리, 저장공간을 효율적으로 사용할 수 있다.
<br></br>

## Human Keypoint Detection (02)

### Convolution Pose Machines

CVPR 2016에서 발표된 CPM 은 completely differentiable 한 multi-stage 구조를 제안했다.

multi stage 방법들은 DeepPose 에서부터 지속적으로 사용되어 왔었지만 crop 연산 등 비연속적인 미분불가능한 stage 단위로 나눠져 있었기 때문에 학습 과정을 여러번 반복하는 비효율적인 방법을 사용해왔다.

+ 참고 : [Convolutional Pose Machines](https://arxiv.org/pdf/1602.00134.pdf)
<br></br>
CPM 은 end-to-end 로 학습할 수 있는 모델을 제안한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/13_wp3QD5J.max-800x600.png)
<br></br>
Stage 1 은 image feature 를 계산하는 역할을 하고 stage 2는 keypoint 를 예측하는 역할을 한다.

g1과 g2 모두 heatmap 을 출력하게 만들어서 재사용이 가능한 부분은 weight sharing 할 수 있도록 세부 모델을 설계 되었다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/14.max-800x600.png)
<br></br>
Stage ≥ 2 에서 볼 수 있듯이 stage 2 이상부터는 반복적으로 사용할 수 있다.

보통 3개의 스테이지를 사용하며, stage 1 구조는 고정이고, stage 2 부터는 stage 2 구조를 반복해서 추론한다. stage 2 부터는 입력이 heatmap (image feature) 이 되기 때문에 stage 단계를 거칠수록 keypoint 가 refinement 되는 효과를 볼 수 있다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/15.max-800x600.png)
<br></br>
CPM 는 사실 좋은 방법은 아니다. Multi - Stage 방법을 사용하기 때문에 end-to-end 로 학습이 가능하더라도 그대로 학습하는 경우는 높은 성능을 달성하기 어렵기 때문이다.

따라서 stage 단위로 pretraining 을 한 후 다시 하나의 모델로 합쳐서 학습을 한다.

하지만 이는 서비스 측면에서 볼 때 비효율 적이며, 불편한 요소이며, 이를 개선하고 있다.

이러한 단점을 가짐에도 CPM 를 사용하는 이유는 성능때문이다. receptive field 를 넓게 만드는 multi stage refinement 방법이 성능향상에 크게 기여한다고 볼 수 있기 때문이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/16.max-800x600.png)
<br></br>
위 그림에서 주황색 실선은 Tompson 알고리즘이며, CPM 에서 제안한 검정색, 회색 실선이 Detection Rate 측면에서 더 높은 성능을 보이고 있음을 알 수 있다.
<br></br>
### Stacked Hourglass Network

ECCV16 에서는 DeepPose 이후 랜드마크라고 불릴만한 논문이 제안 되었는데 바로 Stacked Hourglass Networks for Human Pose Estimation 이다.

+ 참고 : [논문 : Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/pdf/1603.06937.pdf)
<br></br>

#### Hourglass

**Stacked Hourglass Network** 의 기본 구조는 모래시계 같은 모양으로 만들어져 있으며, Conv layer 와 pooling 으로 이미지 (또는 feature) 를 인코딩 하고 upsampling layer 를 통해 feature map 의 크기를 키우는 방향으로 decoding 한다. feature map 크기가 작아졌다 커지는 구조여서 hourglass 라고 표현한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/17.max-800x600.png)
<br></br>

이 방법은 기존 방법과 2가지 큰 차이점이 존재한다.

바로 feature map upsampling 와 residual connection 이다.

pooling 으로 image의 global feature 를 찾고 upsampling 으로 local feature 를 고려하는 아이디어가 hourglass 의 핵심 novelty 라고 할 수 있다.

Hourglass 는 이 간단한 구조를 여러 층으로 쌓아올려서  (stacked) human pose estimation 의 성능을 향상시켰다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/18.max-800x600.png)
<br></br>
MPII 에서 처음으로 PCKh @ 0.5 기준 90% 를 넘어서는 성과를 보이게 된다.
<br></br>
### Simple Baseline

위 연구들을 보면 2D Human Pose Estimation 에서 (x, y) 를 직접 regression 하는 방법이 heatmap 기반으로 바뀌고 모델의 구조가 바뀌어 가면서 encoder-decoder 가 쌓아져 가는 형태가 완성된 것을 알 수 있다.

성능은 개선되었지만 모델이 다소 복잡해졌다.

이때 다른 시각으로 바라 본 것이 바로 SinpleBaseline 이다. 바로 **"기술자체가 많이 발전했는데 현재의 간단한 모델은 얼마나 성능이 좋을까?"** 으로 바라본 것이다.

+ 참고 : [SimpleBaseline](https://arxiv.org/pdf/1804.06208.pdf)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/19.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/20.max-800x600.png)
<br></br>
위 그림과 같이 아주 간단한 encoder - detector 구조를 설계한다. 놀라운 점은 이렇게 간단한 구조를 가지고 COCO 에서 73.7% 의 mAP 를 달성한다.

ResNet 50 만 사용한 간단한 구조가 SOTA 모델인 Hourglass 보다 뛰어난 성능을 보인 것이다.

참고로 **CPN** 은 이전에 소개한 Convolutional Pose Machine 이 아닌 **Cascaded Pyramid Network** 라는 모델이다. 간단히 설명하면 Skip connection 이 stage 사이에 연결되어 있는 것이다.
<br></br>

### Deep High - Resolution Network (HRNet)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/22.max-800x600.png)
<br></br>
HRNet 은 현재까지도 SOTA 에 가까운 성능을 보이는 모델이다.

Simplebaseline의 1저자가 참여해 연구한 모델이기 때문에 Simplebaseline과 같은 철학을 공유한다.

Stacked hourglass, Casecaded pyramid network 등은 multi-stage 구조로 이루어져 있어서 학습 & 추론 속도가 느리다는 큰 단점이 있지만 하이퍼파라미터를 최적화 할 경우 1 - stage 방법보다 성능이 뛰어나다.

반면 Simplebaseline 과 HRNet은 간단함을 추구하는 만큼 1-stage 를 고수한다. 즉, 간단한 구조에 뛰어난 성능을 자랑한다.

+ 참고 : [HRNet](https://arxiv.org/pdf/1902.09212.pdf)
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/23.max-800x600.png)
<br></br>
HRNet 은 위 그림과 같은 알고리즘을 가진다.

위 그림을 자세히 살펴보면

(a) : Hourglass  
(b) : CPN(cascaded pyramid networks)  
(c) : SimpleBaseline - transposed conv  
(d) : SimpleBaseline - dilated conv

를 나타낸다.

또한 SimpleBaseline 은 다른 알고리즘에 비해 성능은 다소 떨어지지만 구조상 공통정과 차이점이 존재한다.

high resolution → low resolution 인 encoder 와 low → high 인 decoder 구조로 이루어진 공통점을 가지며,

Hourglass 는 encoder 와 decoder 의 비율이 거의 비슷하게 대칭적이다. 반면 Simplebaseline 은 encoder 가 무겁고 (resnet50 등 backbone 사용) decoder 는 가벼운 모델을 사용한다. (a), (b) 는 skip connection 이 있지만 (c) 는 skip connection 이 없다는 차이점이 존재한다.

또한 pooling (strided conv) 할 때 소실되는 정보를 high level layer에서 사용해서 detail한 정보를 학습하기 위해 사용한다는 차이점이 존재한다. (사용할 때 더 좋은 성능을 낸다.)

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/24.png)

HRNet 의 저자도 high → low → high 의 구조에서 high resolution 정보(representation) 을 유지할 수 있는 모델을 어떻게 만들 수 있을지 고민하다가 down sample layer 를 만들고 작아진 layer feautre 정보를 다시 up sampling 해서 원본 해상도 크기에 적용하는 모델을 제안했다.

복잡해보이는 것과 달리 1-stage로 동작하기 때문에 전체 flow를 보면 엄청 간단하며 SimpleBaseline 의 백본인 ResNet 을 HRNet 로만 교체하면 된다.

앞서 다룬 CPM 이나 Hourglass 는 중간 단계에서의 heatmap supervision 이 학습과정에 꼭 필요했는데 HRNet 은 필요가 없다.

HRNet 또한 이전 알고리즘 들과 마찬가지로 heatmap을 regression 하는 방식으로 학습하고 MSE loss 를 이용하는데, 이 부분은 SimpleBaseline 과 흡사하다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/25.max-800x600.png)
<br></br>

위 그림으로 결과를 살펴보면 HRNet 이 4% 에 가까운 성능 향상을 보여 줌을 알 수 있다.

+ 참고 : [원저자의 HRNet PyTorch 코드](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
<br></br>

## 코드로 이해하는 Pose Estimation

앞서 살펴본 모델 중 SimpleBaseline 모델을 코드로써 이해해보자.
<br></br>
### SimpleBaseline 구조

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/26.png)
<br></br>

위 그림은 SimpleBaseline 의 구조이다. encoder 역할을 하는 conv layers 와 decoder 역할을 하는 deconv module + upsampling 으로 이루어져있다.

+ 참고 : [논문 : Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/pdf/1804.06208.pdf)
<br></br>

논문을 보고 conv layer 와 deconv module 는 어떤 구성을 가지는지 확인해보자.

conv layer 로는 ResNet 을 사용하며 deconv module 로는 3 단계를 이루고 있으며, deconv 는 256 filter size, 4 x 4 kernel, stride 2 로 2 배씩 feature map 이 커진다.

마지막 출력 레이어는 k 개의  1 x 1 conv layer 로 구성된다.

논문에서보다 논문의 원 저자가 제공하는 공식 코드를 보고 이해해보자.

+ 참고 : [원 저자의 PyTorch 소스 코드 모델 부분](https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
<br></br>
`nn.` 표현이 많이 등장힌다. `torch.nn` 으로 `keras.layers` 와 같이 딥러닝 모델 구성에 필요한 도구들이 정의되어 있다.

29번째 줄에서는 BasicBlock 이라는 클래스가는 keras.models 로 모델을 선언하는 것과 비슷하다.

- 참고로 pytorch model 에서는 사용된 layer 를 forward 함수를 통해 computational graph 를 그려준다.

```python
		residual = x 

		out = self.conv1(x) 
		out = self.bn1(out) 
		out = self.relu(out) 

		out = self.conv2(out) 
		out = self.bn2(out) 

		if  self.downsample is not None: residual = self.downsample(x) 

		out += residual 
		out = self.relu(out)
```
<br></br>
forward 함수를 보면 residual block 를 사용함을 알 수 있다.

Pose 메인 model 을 보면 4 개의 residual block 를 이용하며, 이는 ResNet 과 동일한 것을 알 확인할 수 있다.

흐름을 쉽게 알기 위해 forward 함수를 살펴보자
<br></br>
```python
def forward(self, x): 
	x = self.conv1(x) 
	x = self.bn1(x) 
	x = self.relu(x) 
	x = self.maxpool(x) 
	
	x = self.layer1(x) 
	x = self.layer2(x) 
	x = self.layer3(x) 
	x = self.layer4(x) 

	x = self.deconv_layers(x) 
	x = self.final_layer(x) 

	return x
```
<br></br>
resnet 을 통과한 후 `deconv_layers` 와 `final_layer`를 차례로 통과함을 알 수 있다.

deconv layer 를 보면
<br></br>
```python
		layers.append(nn.ConvTranspose2d(  in_channels = self.inplanes,  out_channels=planes,  kernel_size=kernel,  stride=2,  padding=padding,  output_padding=output_padding,  bias=self.deconv_with_bias))
		  layers.append(nn.BatchNorm2d(planes,  momentum=BN_MOMENTUM))  
		  layers.append(nn.ReLU(inplace=True))
```
<br></br>
transpose conv 와 bn, relu 로 이루어져있는 것을 확인 할 수 있다.

https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml#L23 에서 repo 내에서 검색해보면 세세한 파라미터 관련 정보를 담고 있는 아래 파일을 찾아 deconv layer 의 상세한 파라미터를 확인할 수 있다. 파라미터는 다음과 같다.
<br></br>
```python
NUM_DECONV_LAYERS:  3  
	NUM_DECONV_FILTERS:  
	-  256  
	-  256  
	-  256  
	NUM_DECONV_KERNELS:  
	-  4  
	-  4  
	-  4
```
<br></br>

파이토치로 구성된 코드를 텐서플로우를 활용해 만들어 보자.
<br></br>
> 사용할 라이브러리 임포트 및 tf - SimpleBaseline 모델 선언
```python
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

resnet = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet')
```
<br></br>
> deconv module 생성
```python
upconv1 = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding='same')
bn1 = tf.keras.layers.BatchNormalization()
relu1 = tf.keras.layers.ReLU()
upconv2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding='same')
bn2 = tf.keras.layers.BatchNormalization()
relu2 = tf.keras.layers.ReLU()
upconv3 = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding='same')
bn3 = tf.keras.layers.BatchNormalization()
relu3 = tf.keras.layers.ReLU()
```
<br></br>
> deconv module 에서 중복 제거
```python
def _make_deconv_layer(num_deconv_layers):
    seq_model = tf.keras.models.Sequential()
    for i in range(num_deconv_layers):
        seq_model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding='same'))
        seq_model.add(tf.keras.layers.BatchNormalization())
        seq_model.add(tf.keras.layers.ReLU())
    return seq_model

upconv = _make_deconv_layer(3)
```
<br></br>
> 최종 출력 layer 생성
```python
final_layer = tf.keras.layers.Conv2D(17, kernel_size=(1,1), padding='same')
```
<br></br>
> 가상의 192 x 256 이미지를 넣어서 출력이 잘 되는지 확인
```python
def _make_deconv_layer(num_deconv_layers):
    seq_model = keras.models.Sequential()
    for i in range(num_deconv_layers):
        seq_model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding='same'))
        seq_model.add(tf.keras.layers.BatchNormalization())
        seq_model.add(tf.keras.layers.ReLU())
    return seq_model

resnet = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet')
upconv = _make_deconv_layer(3)
final_layer = tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same')

# input :  192x256
# output : 48x64
inputs = keras.Input(shape=(256, 192, 3))
x = resnet(inputs)
x = upconv(x)
out = final_layer(x)
model = keras.Model(inputs, out)

model.summary()

# np_input = np.zeros((1,256,192,3), dtype=np.float32)
np_input = np.random.randn(1,256,192,3)
np_input = np.zeros((1,256,192,3), dtype=np.float32)
tf_input = tf.convert_to_tensor(np_input, dtype=np.float32)
print (tf_input.shape) # TensorShape([1,256,192,3])

tf_output = model(tf_input)

print (tf_output.shape)
print (tf_output[0,:10,:10,:10])
```
