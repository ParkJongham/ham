# CAM (Class Activation Map)


## Explainable AI

실제 신경망 모델을 적용할 때 모델의 가중치를 최정화한다고 모델이 잘 작동한다고 믿어서는 안된다. 해당 모델이 왜 그렇게 작동하고, 그런 선택을 했는지를 알 수 있어야 한다.

이렇게 작동한 근거를 설명할 수 있는 것을 XAI (explainable AI) 라고 한다.


## 이미지 분류 문제 (Image Classification)  Explainable AI


이미지가 입력으로 들어올 때 일반적으로 CNN 의 앞에 위치하는 특성 추출용 백본 네트워크를 통해 이미지의 local feature 를 추출한다. 이렇게 추출된 특성 맵 (feature map)을 fully connected layer 에 통과시키면 logit 값을 구할 수 있고, 이를 활성화함수 소프트맥스에 통과시키면 입력 이미지가 각 클래스에 속할 확률을 얻을 수 있다.

- logit, sigmoid, softmax 의 관계 : 
	- logit 과 sigmoid 는 서로 역함수 관계
	- 2개의 클래스를 대상으로 정의된 logit 을 K 개의 클래스를 대상으로 일반화할 경우 softmax 함수가 유도된다.
	$$t = logit(y) = \frac{y}{1 - y} \\ y = sigmoid(t) = \frac{1}{1 + exp(t)}$$
	- softmax 함수에서 K 를 2로 놓으면 sigmoid 함수가 된다.
	- sigmoid 함수를 K 개의 클래스를 대상으로 일반화하면 softmax 함수가 된다.

이렇게 이진분류 문제에서 딥러닝이 특정 클래스를 정답으로 냈을 때의 근거는 모델의 레이어마다 feature map 을 시각화해서 activation 이 어떻게 되었는지 확인해보는 것으로 추론이 가능하다.


## GAP (Global Average Pooling)

일반적으로 CNN 레이어를 거쳐 특성을 추출하고, 특성맵을 flattening 후 fully connected layer 에 입력해 줌으로써 각 클래스에 따른 logit 을 산출, logit 을 활성화 함수를 통해 최종적으로 추론한다. 

하지만 Network in Network 논문에서는 fully connected layer 대신 GAP 를 사용하고 softmax 활성화 함수를 사용한다. 이를 통해 CNN 레이어의 채널 수는 데이터의 클래스 수에 맞춰 각 클래스에 따른 확률을 얻는다.

이는 특성 맵의 각 채널이 클래스 별 신뢰도를 나타내기때문에 해석에 용이하고 최적화할 파라미터가 존재하지 않기 때문에 overfitting 을 방지할 수 있다.

- GAP (global average pooling) 기법 : 
  	- 매 채널별로 average pooling 을 채널 값 전체에 global 하게 적용하는 기법.

	- 예를들어 크기가 6 x 6 인 3개의 채널을 가진 특성맵에 GAP 를 수행하면 1 x 1 크기에 3개의 채널을 가진 벡터를 얻게된다. 즉, 6 x 6 크기의 특성 맵의 평균 값을 채널별로 산출하는 것이다.


## CAM (Class Activation Map)


Class Activation Map 는 클래스의 활성화 지도로, 마지막으로 도출해낼 클래스가 지도상의 어디에 활성화가 되어있는지를 나타내 주는 지도인 것이다.

CAM 은 CNN 레이어에서 추출한 특성 맵에 대해 GAP 을 적용하고 이를 소프트맥스 레이어 (소프트맥스 활성화 함수를 가지고 bias가 없는 fully connected layer) 를 적용한다.

GAP 를 통해 추출된 특성맵에서 각 클래스에 대한 정보를 요약하고, 소프트맥스 활성화 함수를 통해 각 채널의 가중합을 구하게 되고, 이를 통해 중요도를 결정한다.
즉, 활성화 함수를 통해 가중합을 구하고 이를 통해 특성 맵의 어떤 부분을 활성화 시킬지 결정한다.

이렇게 출력된 특성맵은 CNN 의 출력값 크기와 같게되며 이를 보간 (interpolation) 해 줌으로써 원본 이미지 크기로 확대한다. 이렇게 확대된 것을 CAM 이라 한다.

각 클래스별 점수 $S_c$ 는 다음과 같이 구할 수 있다.
$$S_c = \sum_k w_k^c \sum_{x, y}\sum_k w_k^c f_k(x, y) \\ w_k^c = 각 \ 클래스 \ (c) \ 노드와 \ k \ 번째 \ 채널의 \ x, y \ 요소의 \ 활성화 \ 값 \\ f_k(x, y) = k  \ 번째 \ 채널의 \ x,y \ 요소의 \ 활성화 \ 값$$

이렇게 구한 $S_c$ 를 통해 Class Activation Map 으로 시각화 한다. CAM 을 얻기 위해서 모든 위치의 활성화 정도를 더해주지않고 각 $x, y$ 위치에서 $k$ 개의 채널만 더해주어 위치정보를 남긴다.

이를 통해 얻은 CAM 은 각 위치에서 채널 별 활성화 정도의 가중합인 $M_c(x, y)$ 가 된다. 이 가중합은 모델이 클래스 $c$ 에 대해 각 위치를 얼마나 보고 있는지 그 정도를 나타낸다.
$$M_c(x, y) = \sum_k w_k^cf_k(x, y)$$

CAM 의 경우 활성화 맵을 얻기위해 GAP 를 사용하여 $f_k$ 를 구하고 그 뒤로 fully connected layer 인 $w_k^c$ 를 추가호 붙여줘야 한다. 즉, 모델의 구조가 제한된다.

또한 CNN 레이어의 결과물만을 시각화할 수 있다.


## Grad - CAM (Gradient CAM)

CAM 과 달리 구조를 제한하거나 변경할 필요가 없으며, 분류 문제 외에 다른 태스크에 대해 유연한다. 

모델이 복잡해지면 복잡해지는 만큼 설명하기가 어려운데 반해 Grad - CAM 은 복잡한 태스크에 사용되는 모델에서도 사용될 수 있다는 장점이 있다.

Grad - CAM 은 "Image Classification"과 "Image captioning", "Visual question Answering" 에 적용할 수 있다.

- Image Classification : 이미지에 대한 분류를 하는 태스크

- Image captioning : 이미지에 대한 설명을 만들어내는 태스크

- Visual question answering : VQA 로 불리며, 어떤 질문과 이미지가 주어졌을 때 이에 대한 답변을 내는 태스크

### Gradient 를 통한 Weight Score 계산

Grad - CAM 은 gradieent 를 통해 CAM 을 얻는다.

클래에스에 대해 backpropagation 을 통해 얻은 $k$ 번째 채널의 그래디언트를 통해 가중치 점수 (weight score) 를 구한다. 이를 식으로 표현하면 다음과 같다.

$$\alpha_k^c = \overbrace{\frac{1}{Z} \sum_i\sum_j}^{gloal \ average \ pooling} \\ \underbrace{\frac{\partial y^c}{\partial A_{ij}^k}}_{gradient \ via \ backprop} \\ y = 모델의 \ 출력값 \\ A = 활성화 \ 맵 \\ i, j  = 각각 \ x 축, \ y축 \\ Z = 전체 \ map \ 의 \ 크기 \\ (i = 1,2,3, ..., u / j = 1,2,3, ..., v \ 일때 \\ Z = u * v \ 가 \ 되어 \ 활성화 \ 맵 \ 전체에 \ 대한 \ global \ average \ 를 \ 구하기 \ 위한 \ 분모 $$

$k$ 개의 채널을 가진 활성화 맵에서 각 채널이 특정 클래스를 활성화하는 데 얼마나 중요하게 작용하는가에 대한 가중치 점수를 구할 수 있다.

위 식을 통해 CAM 처럼 weight 파라미터를 사용하지 않는 다는 것을 알 수 있다.

Grad - CAM 은 $k$ 번째 활성화 맵과 가중치를 곱해준 값을 더한 뒤 ReLU 활성화 함수를 통해 클래스 별 Grad - CAM 을 산출한다. 이를 수식으로 표현하면 다음과 같다. $$L_{Grrad - CAM}^c = ReLU\underbrace{(\sum_k a_k^c A^k)}_{linear \ combination}$$

활성화 함수로 ReLU 를 사용하는 이유는 활성화된 영역을 확인할 때 음의 값은 불필요하기 때문이다.


## 약지도학습 (weakly supervised learning)

CAM 을 통해서 클래스와 이미지만을 데이터로 학습할 뿐만아니라 위치 정보까지 얻을 수 있다. 물론 정확한 위치값은 아닌 간접적인 정보지만 이를 활용해서 원하는 정보를 얻어낼 수 있도록 학습하는 모델 방식을 약지도학습 (weakly supervised learning) 이라고 한다.

약지도 학습은 incomplete supervision, inexact supervision, inaccuracy supervision 의 경우를 포괄적으로 일컫는 말이다.
 
- incomplete supervision : 학습 데이터 중 일부에만 라벨이 달려 있는 경우 (예: 개와 고양이 분류 학습시 10000개의 이미지 중 1000개만 라벨이 있는 경우) 이 경우가 일반적으로 말하는 준지도학습과 같은 경우	

- inexact supervision : 학습데이터의 라벨이 충분히 정확하게 달려있지 않은 경우. (예: 개나 고양이를 Object Detection 또는 Semantic Segmentation해야 하는데, 이미지 내에 정확한 bounding box는 주어져 있지 않고 이미지가 개인지 고양인지 정보만 라벨로 달려있는 경우)

- inaccurate supervision : 학습 데이터에 Noise가 있는 경우 (예: 개나 고양이의 라벨이 잘못 달려있는 경우)

특징이 두드러지게 하는 영역의 가중치 점수를 계산할 수 있었다면, 오히려 해당 가중치 점수를 제거해 주면 Image classification 모델에서 해당 클래스에 대한 prediction이 바뀔 수 있다.  그렇게 제거했을 때 prediction이 바뀌도록 하는 가중치 영역을 모으면 한번도 bounding box 라벨을 보지 않고서도 object detection을 해낼 수 있으며 CAM, Grad - CAM, ACoL 은 약지도 학습을 통한 Object detection, Semantic Segmentation 태스크를 수행할 수 있다.

하지만 CAM 을 통해 활성화 맵을 확인해보면 가장자리보다 특징이 주로 나타나는 위치에 중점적으러 활성화되는 것을 볼 수 있다. 이는 Object detection 에서 물체의 전체적인 형태나 윤곽을 정확히 구분하는 것에 적합하지 못하다. 특징이 주로 나타나는 위치에 집중해 학습을 하기 때문이다.


## ACoL : Adversarial Complementary Learning

ACoL은 모델의 학습에는 끝단이 두 브랜치(branch)로 분리한다. 즉, CAM 을 만들기 위해 활용한 특성 맵을 분리하는데 이는 CAM 을 통한 Object detection 의 특징이 주로 나타나는 위치에 집중해 학습하는 단점을 보완할 수 있다.

나뉜 두 브랜치를 통해 너무 높은 점수를 지워주게되고, 이를 통해 주변의 특성을 반영할 수 있게 한다. 

나뉜 두 브랜치를 Classifier A, B 라고 할 때, 특성 맵은 GAP 를 거쳐 Classifier A 를 거치게되며 loss 로 학습된다. 

이때 ACoL 은 일정 값 이상 활성화된 활성화 맵을 제거하므로, Classifier A 를 통해 학습된 결과는 전체적인 이미지를 보고 클래스를 판별하게된다.

반면에 Classifier B 는 Classifier A 에서 크게 활성화된 영역을 지운 활성화 맵에서 분류를 수행한다.

이렇게 2가지 학습을 통해 더 넓은 영역을 판별의 근거로 삼게되며,  쉽게 맞출 수 있는 샘플을 어렵게 다시 한 번 학습하는 Adversarial Complementary Learning 과정을 거치게된다.


## 1 x 1 Conv

CAM을 얻기 위해서 대상이 되는 네트워크에 feed forward를 하고 활성화 맵과 가중치 계산을 다시 해주어야한다.

이는 분류 모델의 feed forward 와 별개의 작업으로 물체 검출을 위한 모델로 사용하기 위해서 별도의 연산이 필요하다는 단점이 있다.

ACoL 에서는 이를 개선하기 위해 커널 사이즈를 1 x 1, 출력 채널의 개수를 분류하고자 하는 클래스 개수를 가진 컨볼루션 레이어를 특성 맙에 적용하고 여기에 GAP 를 적용하는 방법을 사용하고 있다.

이때 컨볼루션 레이어의 출력값은 곧바로 활성화 맵이 되며, 이 활성화 맵과 CAM 을 비교하여 결과를 나타낸다.
