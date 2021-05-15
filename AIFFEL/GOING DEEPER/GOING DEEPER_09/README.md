# 09. 너의 속이 궁금해 - Class Activatrion Map 살펴보기

딥러닝은 모델이 특정 결과값을 왜 내놓았는지 설명할 수 없다는 특징이 있고, 이 때문에 블랙박스 (Block Box) 모델이라고 불렸다. 이로인해 모델을 신뢰할 수 있는지 명확하지 않았으며, 이러한 한계를 극복하고자 XAI (Explainable Artificail Intelligence), 설명 가능한 인공지능 분야가 연구되고 있다.

설명 가능한 인공지능이란 모델의 결과를 도출해낸 과정을 설명할 수 있는, 신뢰성에 대한 답을 찾는 것이다.


## 학습 목표

1.  분류 모델의 활성화 맵을 이해합니다.
2.  다양한 활성화 맵을 구하는 방법을 알아갑니다.
3.  약지도학습(weakly supervised learning)을 이해합니다.

<br></br>
## Explainable AI

많은 AI 들을 현실에서 서비스, 혹은 문제 해결에 적용할 때 모델의 가중치 (weight) 를 최적화하는 과정만 믿고 모델이 잘 작동한다 (올바그리게 작동한다) 고 확신하기는 어렵다. 

+ 참고 : [유투브 : XAI](https://www.youtube.com/watch?v=U43fxbC-4JQ)
<br></br>

### 이미지 분류 문제 (Image Classification)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-2.max-800x600.png)
<br></br>

위 그림과 같은 이미지 분류모델에 입력으로 이미지가 들어올 경우 일반적인 이미지 분류모델은 이미지의 local feature 를 추출하기 위해 CNN 으로 구성된 특성 추출용 백본 네트워크가 앞에 있다.

백본 네트워크에서 추출된 특성 맵을 Fully Connected Layer 에 통과시켜 얻어진 Logi 을 소프트맥스 활성화 함수에 통과시켜 입력 이미지가 각 클래스에 속할 확률을 얻는다.

+ 참고 : [Logit, Sigmoid, SoftMax 함수의 관계](https://opentutorials.org/module/3653/22995)
<br></br>

Logit 은 Sigmoid 의 역함수이며, Softmax 는 Sigmoid 를 K 개 클래스로 일반화한 것이다. 이를 수식으로 표현하면 다음과 같다.

$$t = logit(y) = \frac{y}{1-y}$$
$$y = sigmoid(t) = \frac{1}{1+exp(t)}$$

이렇게 이미지 분류 모델에서 Logit 값이 K 개 클래스로 일반화한 것을 알 수 있다.

이미지 분류 모델에서 얻어진 Logit 값이 K 개의 클래스 중 하나를 정답으로 가르킬 때, 이를 확인해 볼 수 있는 방안으로는 레이어마다 Feature Map 을 시각화하여 Activation 이 어떻게 되었는지 확인해보는 방법이 있다.
<br></br>

## CAM : Class Acitvation Map

CAM (Class Activation Map) 이란 이미지 분류 분야에서 모델이 어떤 부분을 보고 특정 클래스임을 유추하는지 확인해볼 수 있는 지도를 의미한다.

CAM 을 얻기우해 GAP 를 사용하며, 이를 통해 어느 영역에 의해 Acticvation Map 이 활성화 되었는지를 확인한다.

+ 참고 : 
	+ [논문 : Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
	+ [유투브 : What the CNN is looking](https://youtu.be/fZvOy0VXWAI)
<br></br>

### GAP (Global Average Pooling)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/max_avg_pooling.max-800x600.png)
<br></br>
위 그림과 같이 Pooling 의 종류는 Max Pooling, Average Pooling 이 있다.

Max Pooling 은 각 커널에서 최대값을 찾아 뽑아내는 것이며, Average Pooling 은 각 커널에서 평균값을 찾아내 뽑아내는 것이다. 즉, 커널이 겹치는 영역에 대해 최대값, 평균값을 취한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-4.max-800x600.png)
<br></br>
GAP (Global Average Pooling) 은 매 채널별로, 위에서 보았던 average pooling 을 채널의 값 전체에 global 하게 적용한다.

위 그림과 같이 크기가 6 x 6이고 채널이 3 개인 특성맵에 대해서 GAP 을 수행하면 각 채널이 딱 한 개의 숫자로 요약되어, 1 x 1 크기에 채널이 3 개인 벡터를 얻게 되며, 결과 벡터의 각 차원의 값은 6 x 6 크기의 특성 맵을 채널 별로 평균을 낸 값이 된다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-3.max-800x600.png)
<br></br>
분류 모델의 마지지막 부분에서 Fully Connected Layer 대신 사용하며, 위 그림과 같이 GAP 연산을 한 후 이에 해대 소프트맥스 활성화 함수를 적용한다. 이때 마지막 CNN 레이어의 채널 수는 데이터의 클래스 수에 맞춰 각 클래스에 따른 확률을 얻을 수 있도록 한다.

이를통해 특성 맵의 각 채널이 클래스별 신뢰도를 나타내게 되어 해석을 쉽게하게 해주며, fully connected layer과 달리 최적화할 파라미터가 존재하지 않으므로 과적합 (overfitting) 을 방지할 수 있다.

+ 참고 : 
	-   [논문 : Network In Network](https://arxiv.org/abs/1312.4400)
	-   [navisphere.net의 Network In Network리뷰](http://www.navisphere.net/5493/network-in-network/)
	-   [C4W1L09 Pooling Layers](https://youtu.be/8oOgPUO-TBY)
<br></br>

### CAM

CAM (Class Activation Map) 은 클래스가 활성화되는 지도를 의미한다.

일반적으로 CNN 은 커널 윈도우에 따라 특성을 추출하며, CNN 레이어를 거친 특성맵에도 입력값의 위치정보가 유지된다.

이를 활용하여 특성맵 정보를 Object Detection 이나 Segmentation 등의 문제에 사용한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-5.max-800x600.png)
<br></br>
위 그림은 CAM 을 얻을 수 있는 네트워크를 나타낸다.

CAM 을 얻는 과정은 CNN 레이어를 거쳐 뽑아낸 특성맵에 대해 GAP 를 적용하며, 이에 대해 소프트맥스 레이어 (소프트맥스 활성화 함수를 가지고 bias 가 없는 fully connected layer) 를 적용한다.

이렇게 CNN 을 거친 특성맵에서 각 클래스에 대한 정보는 결과값의 여러 채널에 걸쳐 나타나며, GAP 를 통해 각 채널 별 정보를 요약, 소프트맥스 레이어를 통해 이 정보를 보고 각 클래스에 대한 개별 채널의 중요도를 결정한다.

클래스별 소프트맥스 레이어를 통해 각 채널의 가중합을 구하면 각 클래스가 활성화 맵의 어떤 부분을 주로 활성화 시키는지 확인할 수 있다.

이렇게 얻어진 특성맵은 CNN 의 출력값 크기과 같게되며, 이를 보간 (interpolation) 을 통해 원본 이미지 크기로 확대해주면 CAM 을 얻을 수 있다.

이를 위 그림과 비교해 수식으로 살펴보면 $k = 1, 2, ..., n$ 인 $k$ 번째 채널에 대해서,  
$w_k^c$ 는 각 클래스 $(c)$ 노드와 $k$ 번째 채널 사이의 가중치 값이며, $f_k(x, y)$ 는 $k$ 번째 채널의 $x, y$  요소의 활성화 값이며, 위 그림에서는 파란색 네모 박스로 시각화되어있다.

이 두가지를 곱하고 모든 채널과 $x, y$  축에 대해 더해주면 클래스별 점수 $S_c$ 를 구할 수 있으며, 위 그림의 우측 하단의 최종 CAM 으로 시각화되어있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-6.max-800x600.png)
<br></br>
 CAM 을 얻기위해 위에서 점수를 얻기 위해 모든 위치의 활성화 정도를 더해준 것과 달리 각 $x, y$ 위치에서 $k$ 개의 채널만 더해줘 위치정보가 남도록한다.

이를통해 얻어진 CAM 은 각 위치에서 채널 별 활성화 정도의 가중합인 $M_c(x, y)$ 가 되며, $M_c(x, y)$ 는 모델이 클래스 $c$ 에 대해 각 위치를 얼마나 보고있는지 나타낸다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-3-L-7.png)
<br></br>

+ 참고 : 
	- [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
	-   [POD_Deep-LearningCAM - Class Activation Map의 CAM - Class Activation Map리뷰](https://poddeeplearning.readthedocs.io/ko/latest/CNN/CAM%20-%20Class%20Activation%20Map/)
<br></br>

## Grad - CAM

CAM 의 경우 활성화맵을 얻기위해 GAP 를 사용해 $f_k$ 를 구하고 뒤에 Fully Connected Layer $w_k^c$ 를 붙여준다. 그리고 마지막에 CNN 레이어의 결과물만 시각화 할 수 있다. 즉, 모델의 구조가 제한되는 문제가 있는 것이다.

Grad - CAM (Gradient CAM) 이런 모델 구조가 제한되는 문제를 해결하고 다양한 모델의 구조를 해석할 수 있다.

Grad - CAM 을 통해 CNN 기반의 네트워크는 모델 구조를 변경할 필요가 없으며, 분류 문제 외에 다른 태스크들에 대해 유연한 대처가 가능하다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-8.max-800x600.png)
<br></br>

위 그림은 개와 고양이가 있는 이미지에 대해 Guided Backprop, Grad - CAM, Occlussion Map 의 시각화를 비교한 것이다.

Grad - CAM 에서 높은 분별력과 큰 dimension 을 갖는 CAM 을 만드는 것을 중요하게 본다는 것을 알 수 있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-9.max-800x600.png)
<br></br>
위 그림은 Grad - CAM 의 전체적인 구조를 나타낸 것이다. 그림의 왼쪽에서 은 모델 구조를 나타내며, 이미지를 입력으로 CNN 을 거쳐 특성맵을 추출하고, 이후에 태스크에 따른 다양한 레이어를 사용한다.

오른쪽의 "Image Classification" 과 "Image captioning", "Visual question Answering" 은 Grad-CAM 이 적용될 수 있는 다양한 컴퓨터 비전 문제들을 설명한다. 

mage Captioning 은 이미지에 대한 설명을 만들어내는 태스크이며, Visual question answering 은 VQA 라고하는 어떤 질문과 이미지가 주어졌을 때 이에 대한 답변을 하는 태스크이다.

이러한 모델들은 상당히 복잡하며, 따라서 이를 설명하는 것 또한 복잡하다. 하지만 Grad-CAM은 이런 복잡한 태스크에 사용되는 모델에서도 사용될 수 있다는 장점이 있다.

+ 참고 : [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
<br></br>

###  그래디언트를 통한 Weight Score 계산

CAM 에서는 소프트맥스를 가진 fully connected layer 의 가중치를 통해서 어떤 클래스에 대한 각 채널의 중요도 또는 가중치를 얻어낸다. 

Grad-CAM 에서는 그래디언트 (gradient) 를 사용해 각 채널의 중요도 또는 가중치를 얻어낸다. 특정 클래스에 대해 관찰하는 레이어로 들어오는 그래디언트를 구해, 해당 클래스를 활성화하는 데 레이어의 특성맵에서 어떤 채널이 중요하게 작용하는지 알아낸다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-3-L-10.png)
<br></br>
위 식은 클래스에 대해서 backpropagation 을 통해 얻은 $k$ 번째 채널의 그래디언트를 통해 가중치 점수 (Weight score) 를 구하는 식이다.

$y$ 는 모델의 출력값이며, $A$ 는 활성화 맵을 의미한다. $i, j$ 는 각각 $x$ 축, $y$ 축이며, $Z$ 는 전체 맵의 크기를 나타낸다. $i=1,2,...,u$ 이고, $j=1,2,...,v$ 라면  $Z=u∗v$ 가 되어 활성화 맵 전체에 대한 global average 를 구하기 위한 분모가된다.

이 식을 통해 $k$ 개의 채널을 가진 활성화 맵에서 각 채널이 어떤 클래스를 활성화하는데 얼마나 중요하게 작용하는지 가중치 점수를 구할 수 있다.

즉, 가중치를 구하기 위해 CAM 과 같이 별도의 Weight 파라미터를 도입할 필요가 없다는 것을 알 수 있다.

이렇게 구해진 가중치 점수를 활용하여 활성화맵에서 어떤 클래스 위치에 따른 활성화를 보기위해서는 $k$ 번째 활성화맵과 이 가중치를 곱해주고 합한 뒤 ReLU 활성화 함수를 통해 클래스에 따른 Grad - CAM 을 얻는다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-3-L-11.png)
<br></br>
이를 수식으로 표현한 것이 위 수식이다. 위 고양이와 개에 대한 활성화 부분을 나타내는 Grad - CAM 의 (c), (i) 번째 예시이다.

위 수식에서 Grad - CAM 을 계산할 때 마지막에 ReLU 를 사용하는 이유는, ReLU 를 사용함으로써 활성화된 영역을 확인할 때 불필요한 음의 값을 줄여줄 수 있기 때문이다.
<br></br>

## ACoL : Adversarial Complementary Learning

CAM 은 클래스와 이미지만을 데이터로 학습하지만 위치정보까지 얻을 수 있다는 특징이 있었다.

이렇게 직접적으로 정답 위치정보를 주지않아도 간접적인 정보를 활용하여 학습을 하고 원하는 정보를 얻을 수 있도록 모델을 학습하는 방식을 약지도학습 (Weakly Supervised Learning) 라고한다.

CAM, Grad-CAM, ACoL 은 약지도학습 기법을 활용한 물체 검출(object detection)을 수행할 수 있다. 
<br></br>

### 약지도학습 (Weakly Supervised Learning)

딥러닝에는 여러 학습 방법이 있는데, 준지도 학습 (Semi - Supervised Learning) 과 약지도 학습은 비슷하지만 분명히 다르다.

논문 Adversarial Complementary Learning for Weakly Supervised Object Localization 에서 정의된 바에 의하면 다음과 같이 구분된다.

+ **incomplete supervisio** : 학습 데이터 중 일부에만 라벨이 달려 있는 경우 (예: 개와 고양이 분류 학습시 10000개의 이미지 중 1000개만 라벨이 있는 경우) 이 경우가 일반적으로 말하는 준지도학습과 같은 경우

+ **inexact supervision** : 학습데이터의 라벨이 충분히 정확하게 달려있지 않은 경우. (예: 개나 고양이를 Object Detection 또는 Semantic Segmentation해야 하는데, 이미지 내에 정확한 bounding box는 주어져 있지 않고 이미지가 개인지 고양인지 정보만 라벨로 달려있는 경우)

+ **inaccurate supervision** : 학습 데이터에 Noise가 있는 경우 (예: 개나 고양이의 라벨이 잘못 달려있는 경우)

약지도 학습이란 위 3 가지 경우를 포괄적으로 칭하도록 사용된다. 하지만 이번에는 **inexact supervision** 에 해당하는 경우에 대해 알아보자.

일반적으로 이미지 분류용 학습 데이터보다 바운딩 박스 정보까지 정확히 포함해야하는 Object Detection 이나 Semantic Segmentation 을 위한 학습 데이터가 데이터를 구성하는데 필요한 자원이 많이 소요된다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/grad-cam-counterfactual.png)
<br></br>
위 그림은 Grad - CAM 을 통한  Counterfactual Explanation 의 예 이다.

Grad - CAM 을 통해 개와 고양이의 특징이 두드러지게 나타나는 영역의 가중치 점수를 계산하여고, 해당 가중치 점수를 제거해주면 이미지 분류 모델에서 해당 클래스에 대한 예측이 바뀌게 될 수 있다.

이렇게 가중치 점수를 제거했을 때 예측 결과를 바뀌도록하는 가중치 영역을 모으게된다면 바운딩 박스 라벨을 보지 않고도 Object Detection 을 수행할 수 있다는 것이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/grad-cam_segmentation.png)
<br></br>
위 그림은 Grad - CAM 을 통한 Semantic Segmentation 을 수행한 것이며, Grad - CAM 을 이용한 Object Detection 과 같은 방법으로 수행할 수 있음을 보여준다.

다른 예시로는 자율 주행 연구에서도 약지도 학습을 활용한 예를 볼 수 있다.

+ 참고 : [네이버랩스의 이미지기반의 차선변경 알고리즘(SLC)은 무엇을 보면서 판단을 할까?](https://www.naverlabs.com/storyDetail/16)
<br></br>

### Adversarial Complementary Learning

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-13.max-800x600.png)
<br></br>
위 그림은 ACoL 모델의 구조를 나타낸 것이다. 그림을 보면 모델 학습의 끝단에는 두 브랜치 (branch) 로 나뉘는데, 이는 CAM 을 만들기 위해 활용했던 특성맵을 두 가지로 분리한 것이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-14.max-800x600.png)
<br></br>
CAM, Grad-CAM도 Weaky-supervised 방식의 Object Detection, Segmentation 의 가능성을 보여 주었지만, 한 가지 문제가 있다.

바로 CAM 을 통해 본 활성화 맵을 보면 가장자리보다는 특징이 주로 나타나는 위치에 중점적으로 활성화가 되는 것을 볼 수 있는데, Object Detection 은 새의 부리나 동물의 눈, 시계의 숫자와 같이 부분적 특성이 아닌 물체의 전체적인 형태와 윤곽을 정확하게 구분해내는 것이 중요한데, CAM 모델은 특정 부위에 집중해 학습한다는 것이다.

ACoL 은 이러한 한계를 개선한 모델로 CAM 모델이 특정 부위에 집중해 학습하는 것을 막기위해 ACoL 은 브랜치를 두 가지로 둬서 너무 높은 점수를 지워준다. 이를 통해 주변의 특성을 반영할 수 있도록 하였으며, Adversarial, GAN 에서 생성자와 판별자 처럼, 적대적인 학습 방법이라고 논문에서는 말한다.

ACoL 은 위 모델 구조 그림의 주황색 브랜치를 먼저 거치게된다. 특성맵은 GAP 를거쳐 CAM 에서 보았던 소프트맥스 레이어인 `Classifier A` 를 거치게 된다.

이 브랜치는 loss 로 학습되며, ACoL 은 여기서 얻어진 활성화 맵을 적대적인 방법으로 사용한다.

위에서 언급한 너무 높은 점수를 지워주는 것을 통해 일정 값 이상 활성화된 맵을 지우며, 이를 통해 `Classifier A` 는 쉽게 전체적인 이미지를 보고 클래스를 판별할 수 있는 반면 `Classifier B` 는 A 의 CAM 에서 크게 활성화된 영역을 지운 활성화 맵에서 분류를 해야하기 때문에 더 어려운 문제를 푸는 것과 같아진다.

이렇게 두 가지 `Classifier A` 와 `Classifier B` 를 학습시킴으로써 더 넓은 영역을 판별의 근거로 삼을 수 있다.

이 과정을 통해 모델은 쉽게 맞출 수 있는 샘플을 어렵게 다시 한 번 학습하는 과정, 즉, Adversarial Complementary Learning 과정을 거치게되며, 결과적으로 CAM 이 활성화되는 효과를 확인할 수 있다.
<br></br>

### 1 x 1 Conv

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-L-15.max-800x600.png)
<br></br>
CAM 에서는 CAM 을 얻기 위해서 대상이 되는 네트워크에 feed forward 를 하고 활성화 맵과 가중치 계산을 다시 해줘야 한다.

이 과정은 관찰하고자 하는 분류 모델의 feed forward 와 별개의 작업이므로, 물체 검출을 위한 모델로 사용하기 위해서는 모델의 feed forward 외 별도의 연산을 해주어야 하는 단점이이 있다.

ACoL 논문은 이를 개선하기 위해서 커널 사이즈는 1 x 1, 출력 채널의 개수는 분류하고자 하는 클래스 개수를 가진 컨볼루션 레이어를 특성 맵에 사용하고 여기에 GAP 를 적용하여 Network in Network 에서 본 구조와 유사한 방식을 사용한다. 여기서 컨볼루션 레이어의 출력값은 활성화맵이된다. 이렇게 구해진 활성화 맵과 CAM을 비교한 결과를 위 그림의 왼쪽에서 볼 수 있다.

+ 참고 : -   [Adversarial Complementary Learning for Weakly Supervised Object Localization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.pdf)
<br></br>


