# 05. Object Detection

Object Detection 이란 이미지 내에 존재하는 물체의 위치를 감지하고 해당 물체가 무엇인지 찾아내는 것이다.

## 학습 목표

1.  딥러닝 기반의 Object detection 기법을 배워갑니다.
2.  Anchor box의 개념에 대해 이해합니다.
3.  single stage detection model과 two stage detection 모델의 차이를 이해합니다.

## 실습 디렉토리 구성

```bash
$ mkdir -p ~/aiffel/object_detection/images
```
<br></br>

## 용어 정리

### Object Localization

![출처:https://medium.com/@prashant.brahmbhatt32/the-yolo-object-detection-c195994b53aa](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-2.localization.max-800x600.png)
<br></br>
Object Localization 은 Object Detection 과 달리 이미지 속에 하나의 물체가 있을 때 물체의 위치 정보만을 출력해준다,

주로 Boungding Box 와 Object Mask 를 사용한다.

+ 참고 : 
	 -   [딥러닝 객체 검출 용어 정리](https://light-tree.tistory.com/75)
	-   [Review of Deep Learning Algorithms for Object Detection](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)
<br></br>

### 바운딩 박스 (Bounding Box)

![출처: http://research.sualab.com/introduction/2017/11/29/image-recognition-overview-2.html](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-3.bounding_box.max-800x600.jpg)
<br></br>
바운딩 박스는 이미지 내 존재하는 물체의 위치를 사각형으로 감싼 후 사각형의 꼭지점 좌표로 표현하는 것이다.

이때 꼭지점을 출력하는 2가지 방법이 있다.

하나는 전체 이미지의 좌측 상단을 원점으로 정하고 바운딩 박스 좌상단 좌표와 우하단 좌표 두 가지 좌표, 즉 절대 좌표로 표현한다.

다른 한 가지 방법은 이미지 내의 절대 좌표가 아닌 바운딩 박스의 폭과 높이로 정의하는 방식이다. 이 경우 좌측 상단 점에 대한 상대적인 위치로 물체의 위치를 정의한다.
<br></br>
> 바운딩 박스를 그려볼 이미지 다운
```bash
$ wget https://aiffelstaticprd.blob.core.windows.net/media/images/person.max-800x600.jpg 
$ mv person.max-800x600.jpg ~/aiffel/object_detection/images/person.jpg
```
<br></br>
> 특정 이미지에 바운딩 박스를 그려보자
```python
from PIL import Image, ImageDraw
import os

img_path=os.getenv('HOME')+'/aiffel/object_detection/images/person.jpg'
img = Image.open(img_path)

draw = ImageDraw.Draw(img)
draw.rectangle((130, 30, 670, 600), outline=(0,255,0), width=2)

img
```
<br></br>

### IoU (Intersection over Union)

![[출처: http://research.sualab.com/introduction/2017/11/29/image-recognition-overview-2.html]](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-4.iou.max-800x600.jpg)
<br></br>
바운딩 박스롤 통해 이미지 내 특정 물체의 위치를 인식한 모델은 어떻게 그 성능을 평가할 수 있을까?

각 좌표값의 차이를 L1 이나 L2 로 정의할 수도 있겠지만 이는 박스가 크면 그 값이 커지고 작아지면 그 값이 작아져 크기에 따라 달라지는 문제가 발생할 수 있다.

각 좌표값의 차이와 같이 바운딩 박스 면적의 영향에 받지 않도록 두 개 박스의 차이를 상대적으로 평가하기 위한 방법 중 하나는 IoU (Intersection over Union) 이다.

위 그림와 같이 빨간색 영역 $A_p$는 예측 (prediction) 과 정답 $A_p$ (ground truth) 의 교집합인 영역이고 회색 영역이 합집합인 영역일 때, IoU는 빨간 영역을 회색 영역으로 나눠준 값이 된다.

이때 실제 위치 값과 예측한 위치 값이 일치한다면 IoU 는 1 이 된다.
<br></br>

## Localization

### Target Label

물체의 영역을 바운딩 박스를 통해 숫자로 표현할 수 있으니 Localization 모델을 배워보자.

분류 모델을 만들 때, 컨볼루션 레이어로 구성된 백본 네트워크를 통해 이미지의 특성을 추출한 후 클래스 간 분류를 위한 Fully Connected Layer 를 추가했다.

즉, 표현해야할 클래스에 따라 최종 출력 노드의 갯수가 정해진다.

Localization 을 수행하는 모델은 분류 모델에 output 노드 4 개를 컨볼루션 레이어로 구성된 백본 네트워크 뒤에 추가해 준다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-4-L-5.localization_output.jpg)
<br></br>

위 그림은 위치를 표현할 4개의 노드 라벨이다. 

$p_c$는 물체가 있을 확률이고 물체가 있을 때 $c_1, c_2, c_3$는 각각 클래스 1, 2, 3에 속할 확률이 되게 됩니다. $p_c$가 0 일 경우는 배경인 경우이다. 필요에 따라서는 $c_1, c_2, c_3$ 와 $p_c$ 를 분리하여 다른 활성화 함수를 적용하고 손실을 계산할 수 있다.

$b_x, b_y$ 는 좌측상단의 점을 표현하는 $x$ 축과 $y$ 축의 좌표이고 $b_h$ 와 $b_w$ 는 바운딩 박스의 높이와 폭이다.

이때 $b_x, b_y, b_h, b_w$ 는 모두 입력 이미지의 너비 $w$, 높이 $h$로 각각 Normalize 된 상대적인 좌표와 높이 / 폭으로 표시된다.

+ 참고 : [참고 자료: C4W3L01 Object Localization](https://youtu.be/GSwYGkTfOKk)
<br></br>
> localization 모델을 keras 로 생성
```python
import tensorflow as tf
from tensorflow import keras

output_num = 1+4+3 # object_prob 1, bbox coord 4, class_prob 3

input_tensor = keras.layers.Input(shape=(224, 224, 3), name='image')
base_model = keras.applications.resnet.ResNet50(
    input_tensor=input_tensor,
    include_top=False,
    weights='imagenet',
    pooling=None,
)
x = base_model.output
preds = keras.layers.Conv2D(output_num, 1, 1)(x)
localize_model=keras.Model(inputs=base_model.input, outputs=preds)

localize_model.summary()
```
위 모델에서는 그림의 y 가 target label 로 제공된다.

이때 target label y 는 Ground Truth가 되는 bounding box 의 좌표 (position) 은 위 그림에서와 같이 x1, y1, x2, y2, 입력 image의 너비는 w, 높이는 h로 주어질 때 Bounding box가 x1, y1, x2, y2이고 image의 크기가 w, h 일때
Target label y = [1, x1 / w, y1 / h, (y2 - y1) / h, (x2 - x1) / w] 로 계산할 수 있다.
<br></br>

## Detection (01). 슬라이딩 윈도, 컨볼루션

이제 Object Detection 에 대해 알아보자. Localization 을 수행하는 모델은 입력값으로 들어온 이미지 내 특정 물체가 있는지 확인하고 있다면 물체의 위치를 찾는다.

하지만 1 개의 물체가 아닌 이미지 내 존재하는 여러개의 물체를 찾아와야한다.

이렇게 여러 개의 물체는 어떻게 찾아낼 수 있을까?
<br></br>
### 슬라이딩 윈도우 (Sliding Window)

슬라이딩 윈도우는 입력으로 들어온 이미지를 적당한 크기의 영역으로 나눈 후 각각의 영역에 대해 Localization 을 반복적으로 수행한다.

이미지를 적당한 크기의 영역으로 나눌 때 그 크기를 윈도우 크리하고 하며, 이를 이동시키면서 Localization 을 수행한다.

+ 참고 : [유투브 : Object Detection](https://youtu.be/5e5pjeojznk)
<br></br>

하지만 슬라이딩 윈도우 방식은 이미지가 커지거나 윈도우 사이즈에 따라 계산해야할 양이 많아지면 그만큼 속도가 저하된다. 또한 물체의 크기가 다양해지면 단일 크기의 윈도우로 커버할 수 없다는 단점이 존재해 주로 사용되는 방법은 아니다.

<br></br>

### 컨볼루션 (Convolution)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-7.localization_conv.max-800x600.png)
<br></br>
슬라이딩 윈도우의 속도 문제를 개선하기 위한 방법 중 하나는 슬라이딩 윈도우 대신 컨볼루션을 사용하는 것이다.

컨볼루션은 위 그림에서 14 X 14 크기의 입력에 대해 convolution을 수행했을 때 최종적으로 얻어지는 1 X 1 사이즈의 출력을 sliding window 영역의 localization 결과라고 해석한다면, 거꾸로 14 X 14 크기의 receptive field가 바로 sliding window 영역이 된다.

또한 컨볼루션은 슬라이딩 윈도우와 달리 순차적 연상이 아닌 병렬 연산이 가능하므로 속도면에서 훨씬 빠르다.

위 그림에서는 입력 이미지가 들어왔을 때 특성을 추출하기 위한 첫번째 스테이지를 거친 후 classifier 로 넘어간다. 이때 14 x 14, 16 x 16 크기의 입력 두 가지를 볼 수 있는데 14 x 14에서는 1칸에 대한 결과를 볼 수 있지만 16 x 16에서는 윈도우가 4개가 되어 4칸에 대한 결과를 볼 수 있다.

만약 차량을 찾는 Localization 모델이라면, 차량의 존재 여부에 대한 확률과 바운딩 박스를 정의하기 위한 4 개의 값으로 크기가 5 인 1 x 1 벡터가 의 1 x 1 한칸에 들어가게 됩니다.

또한 16 x 16 크기의 입력 이미지에서 output 은 4 개의 바운딩 박스를 얻을 때 바운딩 박스의 좌표는 서로 다른 원점을 기준으로 학습된다. 

윈도우가 슬라이딩함에 따라 윈도우 내의 물체 위치는 바뀌며, 이때 물체를 표현하기 위한 바운딩 박스의 원점은 윈도우의 좌측 상단이 원점이 된다.
 
+ 참고 : 
	+ [라온피플 머신러닝 아카데미 - Fully Convolution Network](https://m.blog.naver.com/laonple/220958109081)
	+ [유투브 : # Convolutional Implementation Sliding Windows](https://youtu.be/XdsmlBGOK-k)
<br></br>

## Detection (02). 앵커 박스, NMS

### 앵커 박스 (Anchor Box)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-8.sampe_had9R5p.max-800x600.jpg)
<br></br>
위 그림 처럼 이미지 내 차와 사람이 겹쳐있는 경우 어떻게 물체를 인식할 수 있을까?

앞서 배운 슬라이딩 윈도우나 컨볼루션 방법으로는 하나의 물체만 특정할 수 있기 때문에 위 그림에서 차 혹은 사람 둘 중 하나만 감지할 수 있다.

이러한 문제를 해결하기 위한 방법으로 앵커 박스 (Anchor Box) 를 사용할 수 있다.

앵커박스는 서로 다른 형태의 물체와 겹친 경우에 대응할 수 있는 방법으로, 위 그림을 볼 때 일반적으로 차는 좌우로 넓고 사람은 위 아래로 길쭉한 특징이 있다.

앵커박스는 이러한 특징을 통해 차 와 사람과 유사한 형태의 가상의 박스를 두 개 정의한다.

모델의 구조 측면에서 본다면, 두 개의 클래스로 확장해 볼 수 있다. 차 와 사람 클래스에 대해 물체를 감치하기 위해서 한 개의 그리드 셀에 대한 결과값 벡터가 물체가 있을 확률, 2 개의 클래스, 그리고 바운딩 박스 4 개로 총 7 개의 차원을 가지게 된다.

따라서 입력값이 16 x 16일때, 이 그림을 2 x 2로 총 4 칸의 그리드로 나누었다고 하면, 결과값의 형태는 7 개의 채널을 가져 2 x 2 x 7 이 된다.

이때 7개의 차원을 한 벌 더 늘려주어 한 개의 물체의 수를 늘려주며, 앵커 박스가 두 개가 될 때, 결과 값의 형태는 2 x 2 x 14 가 된다.

+ 참고 : [유투브 : Anchor Box](https://youtu.be/RTlwl2bv0Tg)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-4-L-9.anchorbox.png)
<br></br>

위 그림은 앵커 박스를 나타낸 것이다. `Anchor box #1`은 사람을 위해 설정한 크기이고 `Anchor box #2`는 차를 위해 설정한 크기이다.

$y$ 의 라벨을 보면 앵커 박스가 2 개가 됨에 따라서 output dimension이 두 배가 된 것을 볼 수 있으며, 각각은 정해진 Anchor box 에 매칭된 물체를 책임지게 된다.

그림 가장 우측에는 차만 있는 경우 사람을 담당하는 `Anchor box #1` 의 $p_c$ 가 0 이 되고 차를 담당하는 `Anchor box #2` 는 $p_c$ 는 1 이 되도록 클래스와 바운딩 박스를 할당하는 것을 볼 수 있다.

한 그리드 셀에서 앵커 박스에 대한 물체 할당은 위에서 배운 IoU 로 할 수 있으며, 인식 범위 내 물체가 있고 두 개의 앵커 박스가 있는 경우 IoU 가 더 높은 앵커 박스에 물체를 할당된다.

+ 바운딩 박스와 앵커 박스의 구분 : 
	-   바운딩 박스 : 네트워크가 예측한 물체의 위치가 표현된 박스로서, 네트워크의 출력이다.

	-   앵커 박스 : 네트워크가 감지해야 할 물체의 shape에 대한 가정 (assumption) 으로서, 네트워크의 입력이다.
<br></br>
### NMS (Non - Max Suppression)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-10.nms.max-800x600.png)
<br></br>
우리가 2 x 2 또는 더 큰 Grid cell 에서 물체가 있는지에 대한 결과를 받게되면 매우 많은 물체를 받게된다.

Anchor box 를 사용하지 않더라도 2 x 2 격자에 모두 걸친 물체가 있는 경우 하나의 물체에 대해 4 개의 Bounding box 를 얻게되며, 겹친 여러 개의 박스를 하나로 줄여줄 수 있는 방법이 바로 NMS (Non - Max Suppression) 이다.

NMS는 겹친 박스들이 있을 경우 가장 확률이 높은 박스를 기준으로 기준이 되는 IoU 이상인 것들을 제거한다.

IoU 를 기준으로 없애는 이유는 어느 정도 겹치더라도 다른 물체가 있는 경우가 있을 수 있기 때문이다.

이때 Non - Max Suppression 은 같은 클래스인 물체를 대상으로 적용한다.

+ 참고 : [유투브 : Nonmax Suppression](https://youtu.be/VAo84c1hQX8)
<br></br>

## Detection Architecture

앞서 알아본 바운딩 박스, 컨볼루션, 앵커박스, NMS 외에 다양한 방법이 존재한다.

딥러닝 기반의 Object Detection 모델은 크게 2 가지로 구분된다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-11.stage_comparison.max-800x600.png)
<br></br>
1. Single - Stage Detector (1 - Stage Detector)
	+ 객체의 검출과 분류, 바운딩 박스 Regression 을 한꺼번에 수행하는 방법으로 속도가 빠르다.

2. Two - Stage Detector (2 - Stage Detector)
	+ 물체가 있을 법한 위치 후보 (proposals) 를 뽑아내는 단계와 실제 물체가 있는지를 분류하고 정확한 바운딩 박스를 구하는 Regression 을 수행하는 단계가 있다.
<br></br>

### Two - Stage Dectector

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-13.r-cnn.max-800x600.png)
<br></br>
Two - Stage Detector 의 대표적인 모델은 R - CNN 이라는 모델이다.

R - CNN 은 물체가 있을 법한 후보 영역을 뽑아내는 Region Proposal 알고리즘과 후보 영역을 분류하는 CNN 을 사용한다.

Proposal을 만들어내는 데에는 Selective search라는 비신경망알고리즘이 사용하며, 후보 영역의 분류와 바운딩 박스의 Regression 을 위해 신경망을 사용한다.

- 참고 : 
	- [유투브 : Region Proposlas](https://youtu.be/6ykvU9WuIws)
	- [R-CNNs Tutorial](https://blog.lunit.io/2017/06/01/r-cnns-tutorial/)
<br></br>

#### Fast R - CNN
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-14.fast-rcnn.max-800x600.png)
<br></br>
R-CNN 의 경우 region proposal 을 selective search 로 수행한 뒤 약 2,000 개에 달하는 후보 이미지 각각에 대해서 컨볼루션 연산을 수행하게 된다. 따라서 한 이미지에서 특성을 반복해서 추출하며, 이는 속도가 필연적으로 느릴 수 밖에 없다.

Fast R - CNN 은 이름에서도 유추할 수 있듯이 R - CNN 의 속도 문제를 개선한 모델로 후보 영역의 분류와 바운딩 박스 Regression 을 위한 특성을 한 번에 추출하여 사용한다.

R-CNN 과의 차이는 이미지를 Sliding Window 방식으로 잘라내는 것이 아니라 해당 부분을 CNN을 거친 특성 맵에 투영해, 특성 맵을 잘라낸다는 것이다.

이미지를 잘라 개별로 CNN 을 연산하던 R-CNN 과는 달리 한 번의 CNN 을 거쳐 그 결과물을 재활용할 수 있으므로 연산 수를 줄일 수 있다.

이때 잘라낸 특성 맵의 영역은 여러 가지 모양과 크기를 가지므로, 해당 영역이 어떤 클래스에 속하는지 분류하기 위해 사용하는 fully-connected layer에 배치 (batch)  입력값을 사용하려면 영역의 모양과 크기를 맞추어 주어야 하는 문제가 발생한다.

논문에서는 RoI (Region of Interest) Pooling 이라는 방법을 통해 후보 영역에 해당하는 특성을 원하는 크기가 되도록 pooling 하여 사용한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-15.r-cnn_fast-rcnn.max-800x600.png)
<br></br>

#### Faster R - CNN

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-16.faster-rcnn.max-800x600.png)
<br></br>
Fast R - CNN 은 속도의 문제는 개선하였지만 Region Proposal 알고리즘에서 병목 현상이 발생한다.

Faster R - CNN 모델은 이 병목 현상과 속도를 더욱 개선한 모델로 region proposal 과정에서 RPN (Region Proposal Network) 라고 불리는 신경망 네트워크를 사용한다.

이미지 모델을 해석하기 위해 많이 사용하는 CAM (Classification Activation Map) 에서는 물체를 분류하는 태스크만으로도 활성화 정도를 통해 물체를 어느 정도 찾을 수 있다. 이처럼 먼저 이미지에 CNN 을 적용해 특성을 뽑아내면, 특성 맵만을 보고 물체가 있는지 알아낼 수 있다.

이때 특성맵을 보고 후보 영역들을 얻어내는 네트워크가 RPN 이다.

후보 영역을 얻어낸 다음은 Fast R-CNN과 동일하다.

+ 참고 : [R-CNNs Tutorial](https://blog.lunit.io/2017/06/01/r-cnns-tutorial/)
<br></br>

### One - Stage Detector

#### YOLO (You Only Look Once)

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-4-L-17.yolo_design.jpg)
<br></br>

YOLO 는 이미지를 그리드로 나누고, 슬라이딩 윈도 기법을 컨볼루션 연산으로 대체해 Fully Convolutional Network 연산을 통해 그리드 셀 별로 바운딩 박스를 얻어낸 뒤 바운딩 박스들에 대해 NMS 를 한 방식이다.

논문을 보면 이미지를 7 x 7짜리 그리드로 구분하고 각 그리드 셀마다 박스를 두 개 regression 하고 클래스를 구분하게하는데, 이 경우 그리드 셀마다 클래스를 구분하는 방식이기 때문에 두 가지 클래스가 한 셀에 나타나는 경우 정확하게 동작하지는 않는다.

하지만 매우 빠른 인식 속도를 자랑한다. 

+ 참고 : 
	- [YOLO, Object Detection Network](http://blog.naver.com/PostView.nhn?blogId=sogangori&logNo=220993971883)
	- [curt-park님의 YOLO 분석](https://curt-park.github.io/2017-03-26/yolo/)
	- [유투브 : YOLO Algorithm](https://youtu.be/9s_FpMpdYW8)
<br></br>
#### SSD (Single - Shot Multibox Detector)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-18.max-800x600.jpg)
<br></br>
CNN 에서 뽑은 특성 맵의 한 칸은 생각보다 큰 영역의 정보를 담게 된다. 여러 컨볼루션 레이어와 pooling 을 거치기 때문에 한 특성 맵의 한 칸은 최초 입력 이미지의 넓은 영역을 볼 수 있게 된다.

YOLO의 경우 이 특성이 담고 있는 정보가 동일한 크기의 넓은 영역을 커버하기 때문에 작은 물체를 잡기에 적합하지 않는 반면 SSD 는 YOLO 의 단점을 해결하고 다양한 크기의 특성 맵을 활용하고자 한 모델이다.

위 그림에서 볼 수 있듯이 SSD 는 다양한 크기의 특성맵으로부터 분류와 바운딩 박스 Regression 을 수행하며, 이를 통해서 다양한 크기의 물체에 대응할 수 있는 detection 네트워크를 만들 수 있다.

+ 참고 : [yeomko님의 갈아먹는 Object Detection 6 SSD: SIngle Shot Multibox Detector](https://yeomko.tistory.com/20)
<br></br>

## Anchor

### Matching

딥러닝 기반 Object Detection 에는 Anchor 를 쓰지 않는 FCOS (Fully Convolutional One-Stage Object Detection) 와 같은 방법도 있지만 많은 방법들이 Anchor를 기반으로 구현된다.

YOLO 와 Faster R - CNN 에서 Anchor 를 기반으로 Loss 를 계산하는 방식에는 두가지 IoU 를 threshold 로 사용한다.

하나는 Background IoU threshold 그리고 다른 하나는 Foreground IoU threshold 이다.

이는 Faster R - CNN 에서 객체와의 IoU 가 0.7이상일 경우 Foreground 로 할당하고 0.3 이하일 경우는 배경(Background) 으로 할당하는 것으로 사용된다.

이때 0.3 과 0.7 사이의 Anchor 들은 불분명한 영역으로 학습에 활용하지 않는다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-19.car_image.max-800x600.jpg)
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-20.anchor_ex1.max-800x600.jpg)
<br></br>
위의 차량을 Detection 하기 위한 Detection model 을 상상해보면 Feature map 의 Grid 한칸마다 다양한 크기 및 Apsect ratio 를 가진 Anchor 를 만들고 이와 차량의 크기를 비교해서 학습을 해야한다.

위의 예시를 보면 Anchor 의 크기가 차량의 박스 크기에 비해 매우 작은 것을 알 수 있으며, 이 경우 차량 영역에 들어간 Anchor box 이라도 교차하는 면적이 작기 때문에 IoU 가 작아 매칭이 되지 않는 것을 알 수 있다.

따라서 탐지하고자 하는 물체에 따라 Anchor Box 의 크기나 Aspect Ratio 를 조정해줘야 한다.

만약 세로로 긴 물체를 주로 탐지해야 하면 세로로 긴 Anchor box 를 많이 만들고 Matching 되는 box 를 늘려야 한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-21.max-800x600.jpg)
<br></br>
물체 인식을 더 잘하는 방법으로는 Anchor 를 더 키우거나 물체에 맞게 anchor 의 aspect raito 를 조정하는 방법, 다양한 크기의 Anchor를 만드는 방법이 있다.

위 그림에서 차량을 감지하기위해서 Anchor 의 크기를 키우고 Aspect raito 를 조정하여 가로로 길게 만들 수 있다.

그리고 Anchor box 와 녹색의 차량 bounding box 간의 IoU 를 계산해서 물체와의 IoU 가 0.3 이하인 경우는 배경으로 그리고 0.7 이상인 경우는 차량으로 할당되고 두가지 모두에 해당하지 않는 경우에는 Loss 가 학습에 반영되지 않도록해야 한다.

위 그림에서 노란색의 경우는 배경으로 할당되며 녹색만 물체로 학습이되며, 회색은 학습에 반영되지 않는다. 이전에 학습했던 것처럼 노란색의 경우에는 배경이므로 물체의 bounding box 를 추정하기 위한 Regression 은 loss 에 반영되지 않는다. 따라서 파란색 anchor box 만이 regression 을 학습할 수 있는 sample 이 된다.

다양한 물체에 Anchor Box 가 걸쳐있는 경우 가장 높은 IoU 를 가진 물체의 Bounding box 를 기준으로 계산이 된다.

그렇다면 Anchor 를 많이 둘 수록 좋은 Object Detection 모델이 되는걸까?

물론 더 좋은 Recall 을 얻을 수는 있지만 적절하지 못한 Anchor 는 잘못된 결과를 초래할 수 있으므로 오히려 성능을 낮출 수 있다. 또한 Anchor 가 늘어난 만큼 컴퓨팅 자원을 더 소요하기 때문에 적절한 Anchor 선정을 해야한다.
<br></br>

#### Bounding Box Regression

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-L-22.bbox_regression.max-800x600.png)
<br></br>
Anchor box 는 물체를 예측하기 위한 기준이된다.

이때 Anchor  box 에서 Bounding box 를 계산하는 방법은 여러가지가 있다.

YOLO v2 부터 Bounding box Regression 을 사용하는데, 각 Anchor box 에 대응되는 네트워크는 $t_x, t_y, t_w, t_h$ 4 가지 output 으로 Bounding box 를 Regression 해서 정확한 box 를 표현한다.

Bounding box 를 예측하기 위해 예측해야 할 것은 bounding box 의 중심점$(b_x, b_y)$, 그리고 $width(b_w)$ 와 $height (b_h)$이다.

그러나 이 값을 직접 예측하는 것이 아니라 위 그림에 있는 수식과 같이 anchor box의 정보 $c_x, c_y, p_w, p_h$ 와 연관지어 찾는 방법을 사용한다.

기존의 Anchor box 위 좌측 상단이 $c_x, c_y$ 이고 width, height 가 $p_w, p_h$ 이다. 이를 얼마나 x축 또는 y축방향으로 옮길지 그리고 크기를 얼마나 조절해야하는지를 예측하여 물체의 bounding box 를 추론하게 된다.
