# 17. 멀리 있지만 괜찮아


## Dlib 얼굴 인식의 문제점

### Face detection, 얼마나 작고 빨라질 수 있을까?


Dlib 라이브러리를 이용해 Face Landmark 를 찾는 방법으로 카메라 스티커앱을 만들어 보았다.

하지만 얼굴을 잘 못찾거나, 동영상을 처리하기에 너무 느린 속도, 얼굴 각도 및 방향, 크기 변화에 취약한 부분 등의 문제가 발생하였다.


### 왜 작아지고 빨라지는게 중요할까?

일반적으로 얼굴 인식을 위해 딥러닝 서버 구동이 필요한 모델은 서버로 이미지를 보내고, 이를 서버에서 처리한다. 이를 위해선 여러가지를 고려해야하는데, 대표적으로 고려해야할 것은 네트워크 비용, 서버 비용, 인터넷 속도의 영향 이다.

만약 핸드폰 인증 수단으로서 사용되는 서비스라면 어떻게 해야할까?

핸드폰에 모델을 올리기 위해서는 weight 가 작은 모델이 관리에 유리하다. 하지만 일반적으로 작은 모델은 성능이 비교적 떨어지기 마련이며, 이를 위해 보완하기란 쉽지 않다.

+ 참고 : 작고 가벼운 모델을 위한 기업들의 연구
	- 카카오 얼굴인식 관련 리서치 글 :  [https://tech.kakaoenterprise.com/63](https://tech.kakaoenterprise.com/63)
	- 네이버 얼굴검출 관련 오픈소스 :  [https://github.com/clovaai/EXTD_Pytorch](https://github.com/clovaai/EXTD_Pytorch)
<br></br>

### 어떻게 빠르게 만들 수 있을까?

여러가지 방법이 있겠지만 대표적으로 sliding window 를 버리는 것, 병렬화, 적은 파라미터 수로 정확한 성능을 가지도록 모델 설계 등이 있다.

+ 참고 : 병렬화의 방법 및 예시와 중요성

	+ Apple 은 coreml 이라는 라이브러리를 지원함
		+ [https://developer.apple.com/documentation/coreml](https://developer.apple.com/documentation/coreml)
		+ [http://machinethink.net/blog/ios-11-machine-learning-for-everyone/](http://machinethink.net/blog/ios-11-machine-learning-for-everyone/)
		+ 사례 : 16core 뉴럴엔진을 넣은 아이폰12  [iPhone 12 Pro 및 iPhone 12 Pro Max](https://www.apple.com/kr/iphone-12-pro/?afid=p238%7Cs3as1Krbs-dc_mtid_209254jz40384_pcrid_472722877628_pgrid_119804248508_&cid=wwa-kr-kwgo-iphone-Brand-Announce-General-)

	+ 안드로이드의 병렬화 : 
		+ ML kit :  [https://www.slideshare.net/inureyes/ml-kit-machine-learning-sdk](https://www.slideshare.net/inureyes/ml-kit-machine-learning-sdk)
		+ tflite :  [https://www.tensorflow.org/lite?hl=ko](https://www.tensorflow.org/lite?hl=ko)
		+ tftitle 가 안될 경우 직접 별렬 프로그래밍으로 pytorch, tensorflow 같은 툴을 제작
	
	+ 병렬화 도구 경험을 묻는 사례가 많이 등장 :
		+  SIMD :  [https://stonzeteam.github.io/SIMD-병렬-프로그래밍/](https://stonzeteam.github.io/SIMD-%EB%B3%91%EB%A0%AC-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D/)
		+ OpenCL :  [https://www.khronos.org/opencl/](https://www.khronos.org/opencl/)
		+ OpenGL ES :  [https://developer.android.com/guide/topics/graphics/opengl?hl=ko](https://developer.android.com/guide/topics/graphics/opengl?hl=ko)
<br></br>

## Single Stage Object Detection

2 - Stage Detector (Single Stage Object Detection) 은 실행속도가 느리다는 단점을 가지고 있다. 따라서 모델의 속도 향상을 위해서 2-stage 방식의 detection 은 좋은 대안이 될 수없다.

즉, 모델이 가벼워지기 위해서는 sliding window 를 버려야 빨라지는 이유이다. 따라서 모델이 가벼워지기 위해서는 1 - Stage 기반을 사용하는 것이 유리하다.

+ 참고 : Object Detection 관련 복습

	-   object detection 모델을 자세히 설명 :  [Object Detection Part 4: Fast Detection Models](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html)
    
	-   single shot object detectors :  [What do we learn from single shot object detectors (SSD, YOLOv3), FPN & Focal loss (RetinaNet)?](https://jonathan-hui.medium.com/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d)
    
	-   위 글의 번역본 :  [https://murra.tistory.com/17](https://murra.tistory.com/17)
<br></br>

## YOLO (01). YOLO v1 의 등장

### YOLO : You Only Look Once, YOLO v1, big wave 의 시작

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-02.max-800x600.png)
[YOLO 발표, 유투브]([https://youtu.be/NM6lrxy0bxs?t=721](https://youtu.be/NM6lrxy0bxs?t=721))

YOLO 의 출현은 당시 전세계 컴퓨터 비전 학계 및 업계에 충격을 안겨준 기술 진보의 대표적인 사례이다.

대형 서버에서도 아닌 학회장에서 실시간으로 데모를 해버린 모델로 Object Detection 이 실시간으로 돌아간다는 것만으로도 충격적인 결과였다.

<br></br>
## YOLO (02). YOLO v1 의 원리

### RCNN 과 YOLO

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-04.max-800x600.png)

RCNN 은 2 - stage detector 의 대표적인 모델이다.
이 RCNN 과 YOLO 의 차이점은 기본 가정에서부터 존재한다. (Region Proposal Network)

RCNN 계열은 물체가 존재할 것 같은 곳을 backbone network 로 표현할 수 있다 (Gird 내에 물체가 존재한다.) 는 것이 기본 가정이다. 

예를 들어 backbone 을 통과하는 7 x 7 특성맵에서 1 픽셀이 1개의 그리드를 의미할 때, 원본 이미지에서 1개의 Grid Box 의 사이즈는 488 / 7 = 64 로 64 x 64 가 된다. 

7 x 7 특성 맵에서 Bounding Box 와 관련된 Bbox 의 개수는 x ( x, y, w, h, confidence) 5 개의 값이며, Class 확률 C 개의 tensor 를 출력하기 때문에 최종 출력 개수는 7 x 7 x (5 x B + C) 가 된다.

하지만 YOLO v1 는 이미지 내의 작은 영역을 나누면 그 곳에 물체가 있을 수 있다 를 기본 가정으로 한다.

<br></br>
### YOLO 의 Gird Cell

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-05.max-800x600.png)

 YOLO 의 목표는 Grid 에 해당하는 물체를 잘 잡아내는 것이다.

위 그림에서 1 개의 그리드 당 2 개의 Bbox 와 20개 클래스를 예측하는 YOLO 를 만들 때, Output Tensor 를 Flatten 했을 때 크기는 7 x 7 x (5 x 2 + 20) , 즉 1470 이 된다.

자전거에 해당하는 그리드가 많은 경우 학습이 잘 되었다면 해당된 모든 그리드는 모두 비슷한 크기로 자전거 Bbox 를 잡는다.

이때 한 물체에 해당하는 Bbox 가 많아진다면 NMS 와 같은 기법을 이용한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-06.max-800x600.png)

Grid Cell 에 속하는 물체를 검출할 책임이 있으며, 1 개의 그리드에 귀속된 Bbox 정보 (x, y, w, h) 의 학습 목표는 Bbox gt 와 최대한 동일하게 학습되어야 한다.

이때 IoU 를 이용할 수 있다.

<br></br>
### YOLO 의 특징

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-07.max-800x600.png)

기존 R - CNN 계열 방법은 검출 속도가 느리다. 또한 Faster R - CNN 은 RPN 후보군을 뽑고 localization, classification 을 수행함에 따라 RPN 에서 300개 영역을 제안하는데 Objectness 의 숫자가 많을 수록 느려진다는 단점이 있다.

하지만 YOLO 는 Region Proposal 방식의 Grid 기반으로 빠른 속도를 장점으로 가진다.

<br></br>
### YOLO 의 Inference 과정

7 x 7 그리드의 마지막 레이어는 7 x 7 x (30) 에서 30 = 5 (x, y, w, h, c) + 5 + 20 (class) 로 이루어진다.

classification 은 P (real | pred) 인 likelihoood 를 사용한다.

또한 confidence score 를 loss 로 만들 때 P (class | object) * P (object) * IoU 로 표현하므로 7 x 7 x 2 개의 class confidence score 로 계산할 수 있다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-10.max-800x600.png)

+ 참고 : [https://www.slideshare.net/TaegyunJeon1/pr12-you-only-look-once-yolo-unified-realtime-object-detection](https://www.slideshare.net/TaegyunJeon1/pr12-you-only-look-once-yolo-unified-realtime-object-detection)
<br></br>

## YOLO (03). YOLO v1 의 성능

### YOLO v1의 loss 함수

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-11.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-12.max-800x600.png)
<br></br>

### YOLO 의 성능

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-13.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-14.max-800x600.png)

aster RCNN 과 큰 차이가 나지 않으면서 속도는 6배 이상 빠른 성능을 보인다.

<br></br>
### YOLO v1 의 단점

YOLO v1 은 각 그리드 셀이 하나의 클래스만 예측 가능하기 때문에 작은 물체에 대한 예측이 어렵다.

또한 bbox 형태가 학습용 데이터를 통해 학습되기 때문에 분산이 너무 넓어 새로운 형태의 bbox 예측이 어렵다.

모델 구조산 backbone 만 거친 특성맵을 대상으로 Bbox 정보를 예측하기 때문에 localization 이 부정확한 단점이 있다.

<br></br>
## YOLO (04). YOLO v2

YOLO v2 는 YOLO 의 다소 아쉬운 정확도를 개선한 모델이다.

### YOLO v2 의 목적

YOLO v2 의 목적은 3 가지로 정리할 수 있다.

Make it better, Do it Faster, Makes us stronger.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-21.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-22.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-23.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-24.max-800x600.png)

+ 참고 : [YOLO 9000 리뷰](https://dhhwang89.tistory.com/136)
<br></br>

### YOLO v2 의 성능 비교

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-25.max-800x600.png)

+ 참고 : [TED 의 YOLO v2 데모 영상](https://www.ted.com/talks/joseph_redmon_how_computers_learn_to_recognize_objects_instantly?utm_campaign=tedspread&utm_medium=referral&utm_source=tedcomshare)

<br></br>
## YOLO (05). YOLO v3

### RetinaNet 의 도발

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-10-L-26.png)

RetinaNet 의 성능을 보일 때 YOLO v2 를 언급하는데 성능 시각화에도 보여주지 않으며 비교를 했다.

실제로도 YOLO v2 보다 성능이 좋아 YOLO v2 의 시대는 끝난 것 처럼 보였다.

하지만 YOLO v3 이 곧이어 출시되면서 RetinaNet 과 정면 비교를 한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-27.max-800x600.png)

<br></br>
### YOLO v3 원리

[https://taeu.github.io/paper/deeplearning-paper-yolov3/](https://taeu.github.io/paper/deeplearning-paper-yolov3/)

<br></br>
## SSD (01). SSD 의 특징

### SSD : Single Shot MultiBox Detector

YOLO 를 통해 1 - Stage 로 Object Detection 이 가능하다는 것이 증명되자 1 - Stage 기반의 Detector 가 많은 발전을 이루었다.

SSD 는 YOLO v1 에서 그리드를 사용함으로서 생기는 단점을 해결할 수 있는 테크닉을 제안한 모델이다.

특징으로는 Image Pyramid 와 Pre - defined anchor box 를 사용한 것이다.

<br></br>
### Image Pyramid

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-30.max-800x600.png)

SSD 모델은 이미지넷으로 Pretrained 된 VGG - 16 을 사용하였으며, VGG 에서 pooling 을 거친 블록은 하나의 이미지 특성으로 사용이 가능하다.

YOLO 에서 7 x 7 특성맵 하나만을 사용했다면 SSD 는 38 x 38, 19 x 19, 10 x 10, 5 x 5, 3 x 3 .... 을 사용한다.

각 특성 맵은 YOLO 관점에서 볼 때 원본 이미지의 그리드 크기를 달리 하는 효과를 가진다.

즉, 5 x 5 특성맵에서 그리드가 너무 커서 작은 물체를 못찾는 문제를 38 x 38 특성맵에서 찾을 수 있다는 단서를 마련하였다.

하지만, Image Feature Pyramid 는 YOLO 대비 최소 특성맵의 개수가 많은 만큼 계산량이 많으며, 38 x 38 특성맵은 box 를 계산하기에 충분히 깊지 않은 네트워크라는 단점을 가진다.

<br></br>
### Workflow

YOLO v1 의 또 다른 단점은 box 정보 (x, y, w, h) 를 예측하기 위한 seed 정보가 없어 넓은 bbox 분포를 모두 학습할 수 없어 성능 저하가 발생할 수 있다는 점이다.

이 단점을 개선하기 위해서는 Faster RCNN 등에서 사용하는 Anchor 를 적용하면 된다.

이 Anchor box 를 SSD 에서는 Default Box 라고 부른다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-31.max-800x600.png)

SSD 의 framework

-   (a) : 이미지, GT 데이터셋
-   (b) : (vgg backbone 에 가까운) fine-grained feature map. 8 x 8 grid 에서 각각의 grid 에 3 개 anchor box  를 적용할 수 있음. 고양이는 크기가 작기 때문에 (a) 의 고양이는 8 x 8 feature map 내 grid 중 1 개의 anchor box 로 부터 학습될 수 있음
-   (c) : 개의 경우 크고 세로로 긴 경향을 보이기 때문에 receptive field 가 넓은 4 x 4 feature map 이 사용됨

Default box 를 위한 scale. 여러 크기의 default box 생성을 위해 아래와 같은 식을 사용한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-10-L-32.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-33.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-34.max-800x600.png)

$\alpha_r ⇒ r, s_k = s-l$

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-35.max-800x600.png)

<br></br>
## SSD (02). SSD 의 Loss 와 성능

### SSD Loss function

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-36.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-37.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-38.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-39.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-40.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-41.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-42.max-800x600.png)

+ [이미지 출처](https://seongkyun.github.io/papers/2019/07/01/SSD/)
<br></br>

### Hard Negative Mining

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-43.max-800x600.png)

<br></br>
### SSD 의 성능

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-44.max-800x600.png)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-10-L-45.max-800x600.png)

<br></br>
### FCOS

**anchor free**

https://hoya012.github.io/blog/ICCV-2019-paper-preview/ https://blog.naver.com/jinyuri303/221876480557

<br></br>
## Face Detection 을 위한 모델들

#### S3FD

https://seongkyun.github.io/papers/2019/03/21/S3FD/  
https://arxiv.org/abs/1708.05237

#### DSFD

https://arxiv.org/pdf/1810.10220.pdf

#### RetinaFace

https://arxiv.org/pdf/1905.00641.pdf
