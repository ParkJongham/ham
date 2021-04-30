# 01. 백본 네트워크 구조 상세분석


딥러닝 논문의 구조를 파악하고, 최신 기법들을 알아보자.

ResNet 을 먼저 살펴보고, ResNet 이후의 여러 시도들 (DenseNet, SENet)  을 살펴보고 네트워크 구조를 최적화하는 NAS (Neural Architecture Search), EfficientNet 에 대해 알아보자.


## 학습목표

1.  딥러닝 논문의 구조에 익숙해지기
2.  네트워크를 구조화하는 여러 방법의 발전 양상을 알아보기
3.  새로운 논문 찾아보기


## 학습내용

1.  딥러닝 논문의 구조
2.  ResNet의 핵심개념과 그 효과
3.  ResNet 이후 시도 (1) Connection을 촘촘히
4.  ResNet 이후 시도 (2) 어떤 특성이 중요할까?
5.  모델 최적화하기 (1) Neural Architecture Search
6.  모델 최적화하기 (2) EfficientNet
7.  직접 찾아보기

<br></br>
## 딥러닝 논문이 구조

컴퓨터 비전 모델 중 하나인 ResNet 논문을 주제로 딥러닝 논문은 어떤 구조로 이루어져 있는지 알아보자.

먼저 ResNet 은 `Residual Block` 이라는 개념을 도입하여 딥러닝 모델의 깊이가 깊어져도 안정적으로 학습하여 모델의 성능을 개선한 모델이다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-1.max-800x600.png)

참고 :
 -   원본 논문:  [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
-   [Andrew Ng교수님의 C4W2L03 Resnets](https://www.youtube.com/watch?v=ZILIbUvp5lk&feature=youtu.be)

논문에서는 잔차 학습을 실험을 통해 효과를 입증한다.

<br></br>
### 논문의 형식적 구조

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-4.max-800x600.png)

<br></br>
논문은 다음과 같은 정형화된 형식을 가지고 있다.

1. 초록 (abstract)
2. 서론 및 관련 연구 (introduction and related work)
3. 본론 및 실험 (Main subject and experiments)
4. 결론 (conclusion)
5. 참고문헌 및 부록 (reference and appendix)

초록은 아이디어를 제안함과 동시에 이 아이디어가 학계에 어떻게 기여할 수 있는지를 요약한다.

서론 및 관련 연구에서는 어떤 문제에 대하여 제안하고자하는 방법과 이에 관한 이론 설명이 나오며, 문제의식과 논리구조를 명확하게 정리, 제시한다. 
또한 제시된 문제에 대해 어떤 시도들이 있었는지를 말한다. 
소제목에 따라 표현방식이 조금 달라질 수 있다.

본론 및 실험 부분에서는 어떤 독창적인 방법을 적용하였으며, 어떻게 모델을 셋팅하였는지에 대한 정보와 모델을 실험하고 이에따른 결과가 나온다. 

결론에서는 연구 내용의 요약과 추가적으로 진행할 연구 방향을 소개한다.

마지막으로 논문의 작성함에 있어 인용한 논문들의 리스트를 소개하고, 부록에 본문에 설명하지 못한 구현이나 추가적인 실험 설명이 포함된다.

즉, 이전까지의 연구가 해결하지 못한 문제의식, 문제를 해결하기 위한 여러 시도들, 논문만의 독창적인 시도, 시도에 따른 차별화된 성과 이렇게 크게 4가지로 요약할 수 있다.

이렇게 논문의 구조는 이후 후속 논문에 인용되면서 새롭게 문제를 발견하고, 이를 개선하게된다. 따라서 논문에서 제시한 독창적인 아이디어가 무엇인지 파악하고, 발생가능한 문제를 고민하는 것이 중요하다.

<br></br>
## ResNet 의 핵심 개념과 그 효과

### ResNet 논문의 문제의식

논문에서는 '딥러닝 모델의 레이어가 깊어질 수록 과연 성능이 향상되는가' 에 대한 의문점을 제시한다.

실제로 레이어를 깊게 쌓게되면 Vanishing / Exploding Gradient 문제가 발생하며, 이는 모델의 수렴을 방해한다.

하지만 이에 따른 여러 대안 (normalized initialization, intermediate normalization layers)이 존재한다.

ResNet 논문에서는 모델의 수렴을 방해하는 문제와 달리 모델이 수렴하고 있음에도 정확도가 올라가는 것이 아닌 일정 수준에서 머물다 감소하는 문제를 제시하며 이를 해결하고자 한다. 논문에서는 `Degradation Problem` 으로 표현하고 있다.

<br></br>
### ResNet 논문이 제시한 솔루션 : Residual Block

논문에서는 모델이 수렴함에도 정확도가 감소하는 문제를 개선하기위해 Residual Block 라는 아이디어를 제시한다.

즉, **잔차함수** 를 적용하고자 하는데, 이는 일종의 지름길 (shortcut connection) 을 통해 레이어가 입력값을 직접 참조하도록 변경한 것이다.

* 블록 (Block) 이란 ? 
	*  딥러닝 네트워크는 레이어를 쌓는 과정에서 일정한 패던이 반복되기도 하는데, 이렇게 반복되는 레이어 구조를 묶고, 각 구조 속에서 조금씩 바뀌는 부분을 변수로 조정할 수 있게끔 만든다.<br></br>
이렇게 같은 패턴을 가진 레이어를 묶은 단위를 블록이라고 한다.<br></br>
일반적으로 딥러닝 모델은 이러한 블록이 쌓여 구현된다.

shortcut connection 은 앞서 입력으로 들어온 값을 네트워크 출력층에 바로 더해준다.

이를 통해 네트워크는 출력값에서 원본 입력을 제외한 잔차 함수를 학습하게된다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-1-L-2.png)

<br></br>
* 잔차 함수 수식 : 

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-1-L-3.png)

<br></br>
레이어가 깊어졌는데도 모델 성능이 떨어졌다...?
이는 기존 모델에 identity mapping 레이어에 추가적인 레이어를 쌓은 모델이 오히려 identity mapping 레이어보다도 성능이 떨어진다는 의미이며, 학습이 제대로 이루어지지 않았다는 의미다.

논문에서는 이 학습해야할 레이어를 $(H (x))$ 를 $(F(x) + x)$ 로 만들면 어떨까? 라는 생각에서 출발한다.

$(x)$ 는 레이어의 입력값이며, $(F(x) + x)$ 로 변경할 경우 $(F(x))$ 에서 Vanishing Gradient 현상이 발생하여 학습이 안되어 zero mapping 이 되더라도 최종 $H(x)$ 는 $x$, 즉 identity mapping 이 되니 성능의 저하를 막을 수 있지않을까? 라고 생각한 것이다.

위 수식은 $H(x)$ 에 입력값 $(x)$ 를 뺀 형태, 즉 잔차가 된다. 이는 레이어가 많이 쌓일 수록 $(F(x)$ 는 0 에 가까워지지만 $(x)$ 가 있으므로 모델 수렴이 문제가 없다고 생각한 것이다. 실제로 실험을 해보니 안정적인 학습을 보였으며, 레이어가 깊어질 수록 성능이 향상되는 것을 발견하였다.

정리하면, Residual 레이어를 $F(x)$ 로 표현할 때, 레이어의 결과는 입력값 $(x)$ 에 대해 $(F(x))$ 가 된다.

여기서 레이어 입력값 $(x)$ 을 더해주면 최종 출력값은  $(F(x) + x)$ 가 된다.

이후 이 출력값은 활성화 함수 ReLU 를 거친다. $(F(x, W_i))$  는 학습되어야 할 residual mapping 으로서 잔차 학습은 이 식을 학습한다.

논문에서는 shortcut connection 을 가진 ResNet 기본 블록을 Residual Block 로 칭하며, ResNet 은 여러개의 Residual Block 으로 구성된다.


### Experiments

논문의 실험을 보면 ResNet 에 추가된 shortcut connection 이 적용된 네트워크와 적용되지 않은 네트워크를 비교함으로써 성능을 입증한다.

shortcut connection 유무와 네트워크 깊이에 따른 경우를 나눠 모델을 구현하며 각 구현된 모델의 성능을 비교한다.

모델의 깊이는 18개 층을 갖는 네트워크와 34개의 층을 갖는 네트워크로 나누었으며, 각각 shorcut connection 을 적용한 모델과 적용하지 않은 모델로 나눠 비교한다.

이때 shortcut connection 이 적용되지 않은 모델을 plain network 라 한다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-5.max-800x600.png)

위 그림의 왼쪽은 일반 네트워크 이며, 오른쪽은 shortcut connection 이 적용된 네트워크의 학습 결과이다. 일반 네트워크의 결과를 보면 두 개로 네트워크가 깊어지더라도 학습이 잘되지 않는 것을 볼 수 있다.

또한 34개 층을 갖는 네트워크가 18개 층을 갖는 네트워크보다 오류율이 높음을 확인할 수 있다.

하지만 오른쪽의 네트워크 결과를 보면 레이어가 깊어져도 학습이 잘 되는 것을 확인할 수 있다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-1-L-6.png)

위 그림은 학습용 데이터셋이 아닌 검증용 데이터셋을 통한 결과이다. 

Top-1 error란 모델이 가장 높은 확률값으로 예측한 class 1개가 정답과 일치하는지 보는 경우의 오류율이며,   Top-5는 모델이 예측한 값들 중 가장 높은 확률값부터 순서대로 5개 class 중 정답이 있는지를 보는 경우이다.

이 수치들은 낮을 수록 좋다.

결과를 보면 일반 네트워크는 레이어가 늘어났음에도 오류율은 높아졌으며 이는 경사소실 (Vanishing Gradient) 로 인해 학습이 잘 되지 않았기 때문이다.

<br></br>
## ResNet 이후 시도 (01). Connection 을 촘촘히

ResNet 을 통해 성능의 개선을 입증했지만 아직 개선할 부분은 남아있다.

DenseNet 논문의 저자들은 ResNet 의 shortcut connection 을 Fully Connected Layer 처럼 촘촘히 가지게 되면 더욱 성능이 개선될 것이라 생각하였다.

<br></br>
### Dense Connectivity

일반적인 컨볼루션 네트워크가 $L$ 개의 레이어에 대해 각 레이어 간 하나씩의 연결, 즉 총 $L$ 개의 연결을 갖는 것과는 달리, DenseNet 의 기본 블록은 $L$ 개의 레이어가 있을 때 레이어 간 $L(L + 1) / 2$ 개의 직접적인 연결 (direct connection) 을 만든다.

이런 연결구조를 dense connectivity 라고 칭하며, $H_l$ 로 표기하고 이를 합성함수 (composite function) 이라고 부른다.

* dense connectivity 수식 : 

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-1-L-8.png)

각 레이어는 이전 레이어의 특성 맵 전체를 입력값으로 받는다.

식에서 $(X0,X1,...Xl−1)$ 은 $0$ 번째 레이어를 거친 특성 맵부터 $l−1$ 번째 레이어를 거친 특성 맵까지를 의미하며, 이들 은 합성함수 H를 거쳐 l 번째 레이어의 출력값이 된다.

이를통해 경사 소실 문제를 개선하고 특성을 **재사용**할 수 있도록 한다.

ResNet은 shortcut을 원소별로 단순히 더해주었던 반면, DenseNet은 하나하나를 차원으로 쌓아서(concatenate) 하나의 텐서로 만들어 낸다.

이전 ResNet 의 connection 에 다른 연산이 없었던 것과 달리, 합성함수 $Hl$ 은 이 텐서에 대해 배치 정규화(batch normalization, BN), ReLU 활성화 함수, 그리고 3 x 3 컨볼루션 레이어를 통해서 pre - activation 을 수행한다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-1-L-preactivation.png)

Pre - activation 은 [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) 논문에서 제시되었으며, 그림와 (b) 와 같이 ReLU 가 컨볼루션 블록 안으로 들어간 것을 의미한다.

* 참고 : [Pre - activation 의 역할](https://m.blog.naver.com/laonple/220793640991)

DenseNet Block 은 feed-foward 를 포함해서 $L(L + 1) / 2$ 개의 connection 을 가진다. 즉, DenseNet 의 첫번째 블록에서는 6개의 레이어를 사용하므로 6 (6 + 1) / 2 개로 21 개의 레이어를 가진다.

<br></br>
### Growth Rate

ResNet 에서는 특성맵을 더해주지만 DenseNet 은 특성맵을 채널 방향으로 쌓아 사용한다.

만약 4개의 채널을 가진 CNN 레이어를 4개의 DenseNet 블록으로 만들때, 입력값의 채널 갯수가 4 인 경우 블록 내 각 레이어의 입력 값은 어떻게 될까?

첫 번째 레이어의 입력값은 채널의 입력데이터를 그대로 받아들이기 때문에 4 가 되며, 두 번째 레이어의 입력값은 입력 데이터의 채널값과, 첫 번째 레이어 출력값의 채널인 4 을 더해 8 이된다. 세 번째 레이어는 입력 데이터의 채널 4 와 첫 번째 레이어 출력값의 채널 4, 그리고 두 번째 레이어 출력값의 채널 4 를 받아 12 개의 특성 맵을 입력 받고, 네 번째 레이어는 같은 방식으로 16 개의 특성 맵을 입력받는다.

즉, 4씩 커지는 것을 볼 수 있는데 이는 레이어가 깊어질 수록 특성맵의 크기가 매우 커지는 것을 볼 수 있다.

이를 제한하기 위한 방법으로 Growth Rate 값을 조정한다. Growth Rate 값을 조정함으로써 증가하게되는 채널 개수를 조절한다.

위에서 CNN 채널 수를 4 로 설정하였는데 이 값이 바로 Growth Rate 이며, 블록 내 채널 개수를 작게 가져가면서 최종 출력값의 특성맵 크기를 조절할 수 있도록 한다.

이외의 방법은 [DenseNet Tutorial](https://hoya012.github.io/blog/DenseNet-Tutorial-1/) 을 보면서 살펴보자.

DenseNet 논문에서는 이미지넷 챌린지에 참가하기 위해 Growth Rate 를 32 로 사용했다고 한다.

입력값인 이미지 데이터에 3 개의 채널이 있고, Dense Block 의 12 개의 컨볼루션 레이어가 있을 때 각 레이어의 입력 채널을 몇 개가 될 지 알아보자.

<br></br>
> 12 개의 컨볼루션 레이어가 있는 Dense Block 의 각 레이어의 입력 채널 개수 파악
```python
# Dense Block내의 각 레이어 output의 channel을 계산하는 함수
def get_channel_list():
    channel_list = []
    # [[YOUR CODE]]
    input_channel = 32
    growth_rate = 32
    for i in range(12) : 
      channel_list.append(input_channel + growth_rate * i)
    return channel_list

get_channel_list()

# 결과가 [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384] 이 나오면 정상적으로 구현된 것입니다.
```
<br></br>
## ResNet 이후 시도 (02). 어떤 특성이 중요할까?

이번에는 SENet 을 살펴보자. SENet 는 Squeeze - and - Excitation Networks 의 줄임말로 일반적인 CNN 은 입력에 대해 컨볼루션 필터를 필터 사이즈에 따라 적용한다. 이때 필터의 개수가 컨볼루션 레이어 출력값의 채널 수가 된다.

하지만 SENet 에서는 채널 발향으로 global average pooling 을 적용하여 정보를 압축, 활용한다. 이를 통해 중요한 채널이 활성화되로록 한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-10.max-800x600.png)

+ 참고 : 
	 -   원본 논문:  [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
	-   [jayhey님의 gitblog SENet(Squeeze and excitation networks)](https://jayhey.github.io/deep%20learning/2018/07/18/SENet/)
	- [Attention and Memory in Deep Learning and NLP by Denny Britz](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
	- [십분딥러닝_12_어텐션(Attention Mechanism)](https://www.youtube.com/watch?v=6aouXD8WMVQ)

<br></br>
### Squeeze

Squeeze 는 특성에서 중요한 정보를 짜내는 과정에서 붙여진 이름이다. 특성맵 채널에서 어떤 채널이 중요한지 알기 위해서는 채널이 가지고 있는 정보를 압축해서 가져와야한다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-10.max-800x600.png)

채널이 가진 정보를 압축하는 방법은 pooling 기법을 사용하게되며, 커널 영역에 대해 최대값만 남기는 Max Pooling, 평균값을 남기는 Average Pooling 이 있다.

$F_sq$ 함수에서 정보를 압축하여 중요한 정보를 가져오는 Squeeze 과정이 발생한다.

$F_tr$ 이라는 컨볼루션 레이어를 거쳐 $H x W x C$ 의 크기를 가진 특성맵 $U$ 가 나오게되며, $U$ 에 Squeeze 를 적용하면 $1 x 1 x C$ 의 크기가 나온다. 

채널별로 1 개의 숫자만 남도록 2D 특성맵 전체에 대한 평균값을 남기는 Global Average Pooling 을 수행하며, 출력된 값은 $1 x 1 x C$ 백터로 채널별 정보를 압축하여 가지게된다.

<br></br>
### Excitate

채널 별 정보를 압축하여 출력했고, 이를 활용해 어떤 채널을 강조해야할지 판단해야 한다.

논문에서는 강조해야할 채널을 Excitation 으로 표현하였다. Excitation 의 수식은 네트워크 그림의 $F_ex$ 와 동일하다.

* Excitation 의 수식 표현 : 
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-12.max-800x600.png)

수식을 살펴보면 $z$ 는 global average pooling 을 적용한 특성, 즉 정보를 압축하여 가져온 것을 의미한다.

이 특성에 $W_1$ 을 곱해주는 linear 레이어를 거치고 ReLU 활성화 함수 $\delta$ 를 거친다.

이후에 $W_2$ 를 곱해주는 linear layer 을 거치고 시그모이드 활성화 함수 $\sigma$ 를 거친다.

마지막으로 시그모이드 활성화 함수를 사용하는 이유는 가장 중요한 채널 1 개만 활성화 되는 것이 아니라 여러 개의 채널들이 서로 다른 정도로 활성화되도록 하기 위함이다.

+ 참고 : [Multi-Label Image Classification with Neural Network | Keras](https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede)

<br></br>
이렇게 계산된 벡터를 기존의 특성맵 채널에 따라 곱해줌으로써 중요한 채널이 활성화 되도록 한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-13.max-800x600.png)

<br></br>
## 모델 최적화하기 (01). Neural Architecture Search

딥러닝 모델의 파라미터를 최정화해 왔듯이 모델 구조 차체도 최적화 할 수 있다.

이렇게 모델의 구조를 최적화하기 위해 네트워크 구조를 탐색하는 것을 아키택처 탐색 (Architecture Search) 이라고 한다.

여러 아키택처 탐색 방법이 있지만 신경망을 사용해 모델의 구조를 탐색하는 방법을 NAS (Neural Architecture Search) 라고 한다.

NASNet 는 NAS 에 강화학습을 적용하여 최적화한 CNN 모델로 직접 아키택처 탐색을 수행하지 않고 쉽게 사용하기 위해 Tensorflow 에서 [pre - trained NASNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/nasnet) 모델을 제공하고 있다. 

<br></br>
### NASNet

일반적으로 머신 러닝에서는 그리드 탐색 (grid search)  등으로 실험과 모델 셋팅 (config) 를 비교하기 위한 자동화된 방법을 사용한다.

그리드 탐색은 모든 가능한 조합을 실험해 보는 것이다. 구성의 종류가 많은 단점이 있어 딥러닝에는 적합하지 않다.

이런 단점을 보완하기 위해 모델을 탐색하기 위해 강화학습 모델이 대상 신경망의 구성 (하이퍼파라미터) 를 조정하면서 최적의 성능을 내도록 방법을 제안했다.

아키텍쳐 탐색을 하는 동안 강화학습 모델은 대상 신경망의 구성을 일종의 변수로 조정하면서 최적의 성능을 내도록 한다.

레이어의 세부 구성, CNN의 필터 크기, 채널의 개수, connection 등이 조정할 수 있는 변수가 되며, 네트워크 구성에 대한 요소들을 조합할 수 있는 범위를 탐색 공간 (Search Space) 이라고 한다.

MNIST 에 최적화할 때 나온 구조는 다음과 같다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-15.max-800x600.png)

<br></br>
### Convolution Cell

탐색공간이 넓다는 것은 살펴봐야할 범위가 넓다는 것이고, 당연히 오랜 시간이 걸린다.

NASNet 에서는 탐색공간을 줄이기 위해 모듈 (cell) 단위의 최적화를 하고 그 모듈을 조합하는 방식을 채택한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-16.max-800x600.png)

<br></br>
ResNet 에는 Residual Block, DenseNet 에는 Dense Block 이라는 모듈이 사용되는데, 논문에서는 이와 유사한 개념을 convolution cell 이라고 부른다.

Convolution cell 은 normal cell 과 reduction cell 로 구분되며 Normal cell 은 특성 맵의 가로, 세로가 유지되도록 stride 를 1로 고정한다. Reduction cell 은 stride 를 1 또는 2 로 가져가서 특성 맵의 크기가 줄어들 수 있도록 한다. 

논문의 모델은 normal cell 과 reduction cell 내부만을 최적화하며, 이렇게 만들어진 convolution cell 이 위 그림의 두 가지이다. 두 가지 cell 을 조합해 것이 최종 결과 네트워크(NASNet) 를 만들었으며, 좀 더 적은 연산과 가중치로 SOTA (state-of-the-art) 성능을 기록했다.

<br></br>
## 모델 최적화하기 (02). EfficientNet

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-17.max-800x600.png)

EfficientNet 은 기존 모델들의 오류율을 뛰어넘을 뿐만 아니라 모델의 크기인 "Number of Parameters" 또한 최적화 된 것을 볼 수 있다.

<br></br>
+ 참고 : 
	- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
	- [hoya012님의 EfficientNet： Rethinking Model Scaling for Convolutional Neural Networks 리뷰](https://hoya012.github.io/blog/EfficientNet-review/)

위 그림에서 빨간색 선이 EfficientNet 모델이며, 아래로 각 점에 따라 이전에 봤던 모델들이 있는 것을 볼 수 있다.

다른 모델들은 정확도를 얻기위해 많은 파라미터 수를 사용한 반면에 EfficientNet 은 작고 효율적인 네트워크를 사용했으며, CNN 을 효율적으로 사용할 수 있도록 네트워크 형태를 조정할 수 있는 `width`, `depth`, `resolution` 세 가지 요소에 집중한다. 

여기서 `width`는 CNN의 채널에 해당하며, 채널을 늘려줄수록 CNN의 파라미터와 특성을 표현하는 차원의 크기를 키울 수 있다. `depth`는 네트워크의 깊이이다.

`resolution`은 입력값의 너비(w)와 높이(h) 크기이며, 입력이 클수록 정보가 많아져 성능이 올라갈 여지가 생기지만 레이어 사이의 특성맵이 커지는 단점이 존재한다.

<br></br>
### Compound Scaling

EfficientNet 은 앞서 말한 `resolution`, `depth`, `width`를 최적으로 조정하기 위해서 앞선 NAS 와 유사한 방법을 사용해 기본 모델 (baseline network) 의 구조를 미리 찾고 고정한다.

모델 구조가 고정되면 효율적인 모델을 찾는 문제가 개별 레이어의 `resolution`, `depth`, `width` 를 조절해 기본 모델을 적절히 확장시키는 문제로 단순화된다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-L-18.max-800x600.png)

<br></br>
그리고 EfficientNet 논문에서는 `resolution`, `depth`, `width` 라는 세 가지 "scaling factor" 를 동시에 고려하는 compound scaling 을 제안한다.

위 식에서 compound coefficient $\phi$ 는 모델의 크기를 조정하기 위한 계수가 된다.

위 식을 통해 레이어의 `resolution`, `depth`, `width` 를 각각 조정하는 것이 아니라 고정된 계수 $ϕ$ 에 따라서 변하도록 하면 보다 일정한 규칙에 따라 (in a principled way) 모델의 구조가 조절되도록 할 수 있다.

논문은 우선 $ϕ$ 를 1 로 고정한뒤 `resolution`과 `depth`, `width` 을 정하는 $α$, $β$, $γ$ 의 최적값을 찾는다.

논문에서는 그리드 탐색으로 $α$, $β$, $γ$ 을 찾을 수 있었고, $α$, $β$, $γ$ , 즉, `resolution` 과 `depth`, `width` 의 기본 배율을 고정한 뒤 compound coefficient $\phi$ 를 조정하여 모델의 크기를 조정한다.

<br></br>
### 컴퓨터 비전의 주요 태스크를 다루는 논문들

컴퓨터 비전의 주요 태스크 중 관심있는 논문을 직접 찾아보자.

+ 참고 : [9 Applications of Deep Learning for Computer Vision](https://machinelearningmastery.com/applications-of-deep-learning-for-computer-vision/)
