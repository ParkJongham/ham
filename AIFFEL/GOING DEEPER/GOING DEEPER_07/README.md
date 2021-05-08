# 물체를 분리하자! - 세그멘테이션 살펴보기

세그멘테이션 (Segmentation) 은 픽셀 수준에서 이미지의 각 부분이 어떤 이미를 가지는 영역인지 분리하는 방법이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/portrait_mode.max-800x600.jpg)
<br></br>

위 그림을 보면 세그멘테이션을 이해하는데 도움이 된다.

세그멘테이션은 쉽게 말해 이미지를 분할하는 기술이며, 위 그림에선 사람과 배경을 분리하고, 분리된 배경에 아웃 포커싱 효과를 준 것이다.

이러한 세그멘테이션 기술은 핸드폰의 인물모드 등 실 생활에 널리 활용되고 있다.

## 실습 목표

1.  세그멘테이션의 방식을 공부합니다.
2.  시맨틱 세그멘테이션 모델의 결괏값을 이해합니다.
3.  시맨틱 세그멘테이션을 위한 접근 방식을 이해합니다.

## 학습 내용

1.  세그멘테이션 문제의 종류
2.  주요 세그멘테이션 모델
    -   FCN
    -   U-NET
    -   DeepLab 계열
3.  세그멘테이션의 평가
4.  Upsampling의 다양한 방법
<br></br>

## 세그멘테이션 문제의 종류

이미지 내에서 영역을 분리하는 접근 방식은 시맨틱 세그멘테이션 (Semantic Segmentation), 인스턴스 세그멘테이션 (Instance Segmentation) 2 가지가 있다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/semantic_vs_instance.png)
<br></br>
위 그림의 좌측 상단 사진은 어떤 물체들이 모여있는 영역을 인식하는 것을 알 수 있으며, 우측 상단 사진은 양들의 하나 하나의 위치를 식별하는 것을 알 수 있다.

하단의 사진도 좌측은 길과 양들, 풀을 분류하는 반면에 우측은 길과 풀, 그리고 양들 하나 하나를 분류한다.

하단의 좌측과 같은 이미지 분류를 시맨틱 세그멘테이션, 우측과 같은 이미지 분류를 인스턴스 세그멘테이션이라고 한다.

<br></br>
### 시맨틱 세그멘테이션 (Semantic Segmentation)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/segnet_demo.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/leaf_segmentation.max-800x600.png)
<br></br>

위 두 사진은 시맨틱 세그멘테이션의 예 이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/u-net.max-800x600.png)
<br></br>
위 그림은 대표적인 시맨틱 세그멘테이션 모델인 U - Net 의 구조이다.

간단히 구조를 살펴보면 입력으로 572 x 572 크기인 이미지가 들어가고 출력으로 388 x 388의 크기에 두 가지의 클래스를 가진 세그멘테이션 맵 (segmentation map) 이 나온다.

두 가지 클래스는 가장 마지막 레이어의 채널 개수가 2 라는 부분에서 확인할 수 있다.

이대 두 가지 클래스를 문제에 따라 달리 정의하면 클래스에 따른 시맨틱 세그멘테이션 맵 (Semantic Segmentation Map) 을 얻을 수 있다.

예를 들어 의료 인공지능에서 세포 사진에서 병이 있는 영역과 정상인 영역 등을 지정해 적용할 수 있다.

또한 세그멘테이션 모델의 출력값은 이미지 분류에 비해 큰 값임을 볼 수 있는데, 세그멘테이션을 위해 이미지 각 픽셀에 해당하는 영역의 클래스 별 정보가 필요하기 때문이다.

+ 참고 : [SegNet 데모를 통하 시맨틱 세그멘테이션 이해](https://mi.eng.cam.ac.uk/projects/segnet/#demo)
<br></br>

### 인스턴스 세그멘테이션 (Instance Segmentation)

인스턴스 세그멘테이션의 특징은 같은 클래스도 각 각 분리하여 세그멘테이션을 수행한다.

위에 양들의 각 각을 분리한 것을 떠올리자.

인스턴스 세그멘테이션은 Object Detection 을 수행하고 검출된 물체를 각 개체별로 시멘틱 세그멘테이션을 수행하는 방법이 있다.

해당 방법을 적용한 대표적인 모델은 Mask R - CNN 이다. Faster R - CNN의 아이디어인 RoI (Region-of-Interest) Pooling Layer (RoIPool) 개념을 개선하여 정확한 Segmentation에 유리하게 한 RoIAlign, 그리고 클래스별 마스크 분리 라는 단순한 두가지 아이디어를 통해, 클래스별 Object Detection 과 시멘틱 세그멘테이션을 사실상 하나의 Task 로 엮어낸 것으로 평가받는 중요한 모델이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-5-L-FasterRCNN-01.png)
<br></br>
위 그림은 RoI Pool Layer 를 나타낸 그림이다. RoI Pool Layer 는 다양한 RoI 영역을 Pooling 을 통해 동일한 크기의 특성맵으로 추출해내는 레이어이다.

이렇게 추출된 특성맵을 바탕으로 바운딩 박스와 물체의 클래스를 추론 해 낸다.

하지만 물체 영역의 정확한 마스킹을 필요로하는 세그멘테이션을 수행할 때 RoI Pool 과정에서 Quantization 이 필요하다는 한계를 지닌다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-5-L-RoIPool-01.max-800x600.png)
<br></br>

위 그림처럼 16 X 16으로 분할 할 때를 생각해보자. 이미지에서 RoI 영역의 크기는 다양한데, 모든 RoI 영역의 가로/세로 픽셀 크기가 16의 배수인 것은 아니기 때문에  그림처럼 가로 200, 세로 145 픽셀의 RoI 영역을 16 X 16 으로 분할된 영역 중 절반 이상이 RoI 영역이 덮히는 곳들로 끼워맞추다 보면, RoI 영역 밖이 어쩔 수 없이 포함되는 경우가 발생하며, 남는 영역이 버려지는 경우도 발생한다.

이는 시멘틱 세그멘테이션의 정보 손실 및 왜곡을 야기한다.

예를 들어 위 그림에서 3 X 3 의 RoI Pool 을 적용한다면 가로 / 세로 각각 3 의 배수 만큼만 영역에 적용되므로 가로는 6, 세로는 3 만큼만 적용된다. 따라서 나머지 부분 (위 그림에서 진한 파랑, 연한 파랑 부분)의 정보를 소실하게 되는 것이며, 일부 불필요한 영역(위 그림에서 녹색 영역 부분) 은 포함되는 경우가 발생한다.

Mask R - CNN 의 RoIAlign 은 Quantization 하지 않고도 RoI 를 처리할 고정 사이즈의 특성맵을 생성할 수 있게 아이디어를 제공한다.

RoI 영역을 pooling layer 의 크기에 맞추어 등분한 후, RoI Pool 을 했을 때의 quantization 영역 중 가까운 것들과의 bilinear interpolation 계산을 통해 생성해야 할 특성맵을 계산 한다는 것이 아이디어의 핵심이다.

+ 참고 : [Understanding Region of Interest — (RoI Align and RoI Warp)](https://towardsdatascience.com/understanding-region-of-interest-part-2-roi-align-and-roi-warp-f795196fc193)
<br></br>

또한 Mask R - CNN은 Faster R - CNN 에서 특성 추출방식을 "RoIAlign" 방식으로 개선을 하고 세그멘테이션을 더한 방식이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/instance_seg.max-800x600.png)
![](https://aiffelstaticprd.blob.core.windows.net/media/images/mask-rcnn-head.max-800x600.png)
<br></br>
위 그림은 Mask R - CNN 과 Faster R - CNN 과 비교한 구조도이다. Mask R - CNN 은 U - Net 처럼 특성맵의 크기를 키워 마스크를 생성해내는 부분을 통해 인스턴스에 해당하는 영역, 즉 인스턴스 맵을 추론하며, 클래스에 따른 마스크를 예측할 때, 여러 가지 태스크를 한 모델로 학습하여 물체 검출의 성능을 높인다.

Bounding box regression을 하는 `Bbox head`와 마스크를 예측하는 `Mask Head`의 두 갈래로 나뉘는 것을 볼 수 있는데, Mask map의 경우 시맨틱 세그멘테이션과 달리 상대적으로 작은 28 x 28의 특성 맵 크기를 가지며, RoI Align 을 통해 줄어든 특성에서 마스크를 예측하기 때문에 사용하려는 목적에 따라서 정확한 마스크를 얻으려는 경우에는 부적합할 수 있다.
<br></br>

## 주요 세그멘테이션 모델 (01). FCN

![](https://aiffelstaticprd.blob.core.windows.net/media/images/fcn.max-800x600.png)
<br></br>
위 그림은 세그멘테이션의 여러가지 방법 중 FCN (Fully Convolutional Network) 이다. 

![](https://aiffelstaticprd.blob.core.windows.net/media/images/fcn_2.max-800x600.png)
<br></br>
AlexNet, VGG - 16 등의 모델을 세그멘테이션에 맞게 변형한 모델로 기본적인 VGG 모델은 이미지 특성을 추출하기 위해 네트워크 뒷 부분에 Fully Connected Layer 를 붙여 계산한 클래스별 확률을 바탕으로 이미지 분류를 수행하는 반면 FCN 는 세그멘테이션을 위해 네트워크 뒷 부분에 Fully Connected Layer 대시 CNN 을 붙여준다.

CNN 을 붙여주는 이유는 CNN 은 이미지 내의 위치 정보를 유지하는 특징이 있기 때문이다. 반면에 Fully Connected Layer 는 위치를 고려하지 않는다.

위치 정보를 유지하면서 클래스 단위의 히트맵을 얻어 세그멘테이션을 하기 위해 CNN 을 사용하는 것이다.

이 CNN 은 1 x 1의 커널 크기 (kernel size) 와 클래스의 개수만큼의 채널을 가지며 이를 CNN 을 거치게해 클래스 히트맵을 얻는다.

하지만 히트맵의 크기는 CNN과 pooling 레이어를 거치면서 크기가 줄었기 때문에 일반적으로 원본 이미지보다 작다. 따라서 이를 원본 이미지 크기로 키워주는 UpSampling 을 적용해야한다.

UpSampling 의 여러 방법 중 FCN 은 Deconvolution 과 Interpolation 방법을 활용한다.

Deconvolution은 컨볼루션 연산을 거꾸로 해준 것이며,  Interpolation 은 보간법으로 주어진 값들을 통해 추정해야 하는 픽셀 (여기서는 특성 맵의 크기가 커지면서 메꾸어야 하는 중간 픽셀들을 의미) 추정하는 방법이다.

Interpolation 보간법은 Linear interpolation과 Bilinear interpolation, 2 가지로 나뉠 수 있는데 Linear interpolation 은 1차원 상의 두 개의 점 사이에서 거리 비에 따라 추정하는 것이며, Bilinear interpolation 은 2차원으로 확장해서 4개의 점 사이에서 어떤 점의 값을 추정하는 것이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/fcn_3.max-800x600.png)
<br></br>
FCN - 32s 는 UpSampling 을 통해 원하는 세그멘 테이션 맵을 얻은 경우이다.

논문에서는 더 나은 성능을 위해 Skip Architecture 라는 방법을 제안하는데, FCN - 32s, FCN - 16s, FCN - 8s로 결과를 구분해 설명한다.

FCN - 16s는 앞쪽 블록에서 얻은 예측 결과 맵과, 2 배로 upsampling 한 맵을 더한 후, 한 번에 16 배로 upsampling 을 해주어 얻는다. 여기서 한 번 더 앞쪽 블록을 사용하면 FCN - 8s 를 얻을 수 있다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/fcn_4.max-800x600.png)
<br></br>

위 그림을 통해 결과를 확인할 수 있다.

+ 참고 :
	-   [Fully Convolutional Networks for Semantic Segmentation - 허다운](https://youtu.be/_52dopGu3Cw)
	-   [FCN 논문 리뷰 — Fully Convolutional Networks for Semantic Segmentation](https://medium.com/@msmapark2/fcn-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-fully-convolutional-networks-for-semantic-segmentation-81f016d76204)
	-   원본 논문:  [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
<br></br>

## 주요 세그멘테이션 모델 (02) U - Net

![](https://aiffelstaticprd.blob.core.windows.net/media/images/u-net.max-800x600.png)
<br></br>
위 그림은 세그멘테이션의 대표적인 모델인 U - Net 이다. 그림에서 볼 수 있듯이 U 자 형태의 구조를 가지고 있다.

U - Net 은 FCN에서 upsampling 을 통해서 특성 맵을 키운 것을 입력값과 대칭적으로 만들어 준 것이며, 세그멘테이션 뿐만 아니라 여러가지 이미지 태스크에서 사용된다.

+ 참고 : 
	-   [딥러닝논문읽기모임의 U-Net: Convolutional Networks for Biomedical Image Segmentation](https://www.youtube.com/watch?v=evPZI9B2LvQ)
	-   [U-Net 논문 리뷰 — U-Net: Convolutional Networks for Biomedical Image Segmentation](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a)
	-   원본 논문:  [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
<br></br>
### U - Net 의 전체 구조

논문에서는 네트워크 구조를 좌측의 Contracting path 와 우측의 Expansive path, 두 가지로 구분한다.

U - Net 구조 그림 우측의 Contracting path 는 일반적으로 우리가 사용해왔던 Convolution network 와 유사한 구조를 가지며, 각 블록은 두개의 3 x 3 convolution  계층과 ReLu 를 가지고 그 뒤로 downsampling 을 위해서 2 x 2의 커널을 2 stride 로 max pooling 을 수행한다. Downsampling 을 거친 후 다음 convolution 의 채널 크기는 두 배씩 늘어나도록 설계되었다.

Expansive path 에서는 각 블록에 2 x 2 up-convolution 이 붙어 채널이 절반씩 줄어들고 특성 맵의 크기는 늘어나며, contracting block과 동일하게 3 x 3 convolution 이 두 개씩 사용 되었다.

두 Path 에서 크기가 같은 블록의 출력과 입력은 skip connection 처럼 연결해주어 low - level 의 feature 를 활용하도록 한다. 마지막에는 1 x 1 convolution 으로 원하는 시맨틱 세그멘테이션 맵을 얻을 수 있다.

결과적으로 입력 크기가 572 x 572 인 이미지가 들어가며, 388 x 388 크기의 두 가지 클래스를 가진 세그멘테이션 맵 이 출력된다.

Convolution 은 padding 을 통해서 크기를 같게 유지할 수 있으나, U-Net 에선 padding 을 하지않아서 deconvolution 으로 확대하더라도 원래 이미지 크기가 될 수 없다. 따라서 입력 이미지의 크기와 출력 이미지가 다른것은 세그멘테이션 맵을 원하는 크기로 조정 (resize) 하여 해결한다.
<br></br>

### 타일 (Tile) 기법

![](https://aiffelstaticprd.blob.core.windows.net/media/images/unet.max-800x600.png)
<br></br>
FCN 은 입력 이미지의 크기를 조정하여 세그멘테이션 맵을 얻어내는 반면 U - Net 은 타일 (Tile) 방식을 사용해 서로 어느 정도 겹치는 구간으로 타일을 나눠 네트워크를 추론한다. 이를 통해 큰 이미지에서도 높은 해상도의 세그멘테이션 맵을 얻을 수 있다.
<br></br>

### 데이터 불균형 해결

![](https://aiffelstaticprd.blob.core.windows.net/media/images/unet_2.max-800x600.png)
<br></br>
만약 세포를 검출한다고 가정하자. 세포를 검출하기 위해서는 세포의 영역뿐만 아니라 세포의 경계 역시 예측해야한다.

세포 간 경계를 픽셀 단위로 라벨을 매긴다고 한다면 데이터셋에 세포나 배경보다 세포 간 경계의 면적은 절대적으로 작을 것이다.

이러한 클래스 간 데이터 양의 불균형을 해결하기 위한 방법으로 분포를 고려한 Weight Map 을 학습 때 사용했다고 한다.

이 Weight 는 파라미터가 아닌 손실함수에 적용되는 가중치를 의미한다.

의료 영상에서 세포 내부나 배경보다는 상대적으로 면적이 작은 세포 경계를 명확하게 추론해 내는 것이 더욱 중요하기 때문에, 세포 경계의 손실에 더 많은 페널티를 부과하는 방식이다.
<br></br>

## 주요 세그멘테이션 모델 (03). DeepLab 계열

DeepLabv3+ 는 이름에서 볼 수 있듯이 이전의 많은 버전을 거쳐 개선을 이뤄온 네트워크 이다.

처음 DeepLab 모델이 제안된 뒤 이 모델을 개선하기 위해 Atrous Convolution 와 Spatial Pyramid Pooling 등 많은 방법들이 제안되어 왔습으며, DeepLabv3+ 의 전체 구조를 본 뒤 Dilated Convolution 이라고도 불리는 Atrous Convolution 과 Spatial Pyramid Pooling 을 살펴보자.
<br></br>
### 전체 구조

![](https://aiffelstaticprd.blob.core.windows.net/media/images/deeplab_v3.max-800x600.png)
<br></br>

위 그림은 DeepLabV3+ 의 구조이다. 위 그림의 인코더 (Encoder), 디코더 (Decoder) 는 U - Net 에서 Contracting path 과 Expansive path 의 역할을 한다.

인코더는 이미지에서 필요한 정보를 특성으로 추출해내는 모듈이고 디코더는 추출된 특성을 이용해 원하는 정보를 예측하는 모듈이다. 

3 x 3 convolution 을 사용했던 U - Net 과 달리 DeepLabV3+ 는 Atrous Convolution 을 사용하며, Atrous Convolution 을 여러 크기에 다양하게 적용한 것이 ASPP (Atrous Spatial Pyramid Pooling) 이다.

DeepLab V3+ 는 ASPP 가 있는 블록을 통해 특성을 추출하고 디코더에서 Upsampling 을 통해 세그멘테이션 마스크를 얻는다.

+ 참고 : 
	-   [Lunit 기술블로그의 DeepLab V3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/)
	-   [hyunjulie님의 2편: 두 접근의 접점, DeepLab V3+](https://medium.com/hyunjulie/2%ED%8E%B8-%EB%91%90-%EC%A0%91%EA%B7%BC%EC%9D%98-%EC%A0%91%EC%A0%90-deeplab-v3-ef7316d4209d)
	-   [Taeoh Kim님의 PR-045: DeepLab: Semantic Image Segmentation](https://www.youtube.com/watch?v=JiC78rUF4iI)
	-   원본 논문:  [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
<br></br>

### Atrous Convolution

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/atrous_conv_2.gif)
<br></br>
Atrous Convolution은 간단히 말하면 "띄엄띄엄 보는 컨볼루션" 이다. 

Convolution은 좌측의 일반적인 컨볼루션과 달리 더 넓은 영역을 보도록 해주기 위한 방법으로 커널이 일정 간격으로 떨어져 있는데 이를 통해 컨볼루션 레이어를 너무 깊게 쌓지 않아도 넓은 영역의 정보를 커버할 수 있게된다.

+ 참고 : [딥러닝에서 사용되는 여러 유형의 Convolution 소개](https://zzsza.github.io/data/2018/02/23/introduction-convolution/)
<br></br>
### Spatial Pyramid Pooling

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-5-L-SPP.max-800x600.png)
<br></br>
Spatial Pyramid Pooling 은 여러 가지 스케일로 convolution 과 pooling 을 하고 나온 다양한 특성을 연결(concatenate) 해준다. 

이를 통해서 멀티스케일로 특성을 추출하는 것을 병렬로 수행하는 효과를 얻을 수 있다.

여기서 컨볼루션을 Atrous Convolution으로 바꾸어 적용한 것은 Atrous Spatial Pyramid Pooling 이라고하며,  이러한 아키텍쳐는 입력이미지의 크기와 관계없이 동일한 구조를 활용할 수 있다는 장점이 있다. 

따라서 제각기 다양한 크기와 비율을 가진 RoI 영역에 대해 적용하기에 유리하다.

+ 참고 : [갈아먹는 Object Detection - Spatial Pyramid Pooling Network](https://yeomko.tistory.com/14)
<br></br>

## 세그멘테이션의 평가

![](https://aiffelstaticprd.blob.core.windows.net/media/images/segmentation_metric.max-800x600.png)

+ 참고 : [Evaluating image segmentation models](https://www.jeremyjordan.me/evaluating-image-segmentation-models/)
<br></br>
일반적으로 시맨틱 세그멘테이션의 결과값은 이미지 크기에 맞는 세그멘테이션 맵 크기와 시맨틱 클래스의 수에 맞는 채널 크기를 가진다. 

각 채널의 Max Probability 에 따라 해당 위치의 클래스가 결정된다.

위 그림에서 Ground truth 와 Prediction 의 정오 여부를 가지를 방법으로는 픽셀 수 만큼의 분류 문제로 평가한다.
<br></br>
### 픽셀별 정확도 (Pixel Accuracy)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/error_metric.max-800x600.jpg)
<br></br>
Pixel Accuracy 는 쉽게 말해서 픽셀에 따른 정확도를 의미한다.

세그멘테이션 문제를 픽셀에 따른 이미지 분류 문제로 생각했을 때, 우리는 마치 이미지 분류 문제와 비슷하게 픽셀별 분류 정확도를 세그멘테이션 모델을 평가하는 기준으로 생각할 수 있다.

이 때 예측 결과 맵 (prediction map) 을 클래스 별로 평가하는 경우에는 이진 분류 문제 (binary classification) 로 생각해 픽셀 및 채널 별로 평가한다. 픽셀 별 이미지 분류 문제로 평가하는 경우에는 픽셀 별로 정답 클래스를 맞추었는지 여부, 즉 True / False를 구분한다.

예를 들어, 4 x 4 의 크기를 가지는 map 에서 중앙의 2 x 2 의 영역이 전경이고 예측 결과 중 한 칸을 놓쳤다면 위에서 보이는 Error Metrics 를 확인하면 (TP + TN) / (FP + FN + TP + TN) 으로 Accuracy 를 구할 수 있다. TP (True positive) + TN (True negative) 는 옳게 분류된 샘플의 수로 잘못 예측된 한 칸을 제외한 15 이다.

그리고 False case는 1 인 한칸은 전경이 배경으로 예측되었으니 FN (False negative) 이며, 따라서 분모항은 16 이 됩니다. 따라서 Pixel Accuracy 는 15 / 16으로 계산할 수 있다.
<br></br>

### 마스크 IoU (Mask Intersection - Over - Union)

물체 검출 모델을 평가할 때는 정답 라벨 (ground truth) 와 예측 결과 바운딩 박스 (prediction bounding box)  사이의 IoU 를 사용한다.

마스크도 일종의 영역임을 생각했을 때 세그멘테이션 문제에서는 정답인 영역과 예측한 영역의 IoU 를 계산할 수 있다.

> 세그멘테이션 마스크의 IoU 를 구하는 식
```python
# sample for mask iou 

intersection = np.logical_and(target, prediction) 

union = np.logical_or(target, prediction) 

iou_score = np.sum(intersection) / np.sum(union)
```
<br></br>
마스크 IoU 를 클래스 별로 계산하면 한 이미지에서 여러 클래스에 대한 IoU 점수를 얻을 수 있으며, 이를 평균하여 전체적인 시맨틱 세그멘테이션 성능을 평가한다.
<br></br>
## Upsampling 의 다양한 방법

### Nearest Neighbor

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/upsampling1.png)
<br></br>
Nearest upsampling 은 scale을 키운 위치에서 원본에서 가장 가까운 값을 그대로 적용하는 방법이다.

위 그림과 같이 2 x 2 행렬이 있을 떄 이를 2 배로 키우면 4 x 4 행렬이 된다.

이때 좌측 상단으로부터 2 x 2 는 입력 행렬의 1 x 1 과 가장 가깝기 때문에 해당 값을 그대로 사용한다.
<br></br>

### Bilinear Interpolation

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/bi_interpolation.png)
<br></br>
Bilinear Interpolation 은 두 축에 대해서 선형보간법을 통해 필요한 값을 메우는 방법이다.

위 그림과 같이 2 x 2 행렬을 4 x 4 행렬로 upsampling 할 때 빈 값을 채워줘야한다.

이렇게 빈 값을 채워주기 위해 선형보간법을 사용하는데, 축을 두 방향으로 활용한다.

위 그림에서 $R_1$ 이 $Q_11$ 과 $Q_21$ 의 $x$축 방향의 interpolation 결과이며, $R_2$ 는 $Q_12$ 와 $Q_22$ 의 $x$ 축 방향의 interpolation 결과이다.

$R_1$ 과 $R_2$ 를 $y$축 방향으로 interpolation 하면 새로운 위치 $P$ 의 값을 추정할 수 있다.

+ 참고 : [bskyvision의 선형보간법(linear interpolation)과 삼차보간법(cubic interpolation), 제대로 이해하자](https://bskyvision.com/m/789)
<br></br>

### Transposed Convolution

![](https://aiffelstaticprd.blob.core.windows.net/media/images/transposed_conv.max-800x600.jpg)
<br></br>
Transposed Convolution 은 파라미터를 가진 Upsampling 방법 중 하나이다.

Convolution Layer 는 Kernel 의 크기를 정의하고 입력된 Feature 를 Window 에 따라서 output 을 계산하는데, Transposed Convolution은 이와 반대의 연산을 한다.

즉, 거꾸로 학습된 파라미터로 입력된 벡터를 통해 더 넓은 영역의 값을 추정한다.

+ 참고 : [zzsza님의 Up-sampling with Transposed Convolution 번역](https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/)
