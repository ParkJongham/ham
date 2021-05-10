# 12. 인공지능으로 세상에 없던 새로운 패션 만들기

### 학습 전제

-   Convolution의 padding, stride 등의 기본 개념을 알고 있다.
-   교차 엔트로피(Cross Entropy) 등의 손실 함수, 최적화 함수 등 딥러닝의 기본적인 학습 알고리즘을 알고 있다.
-   텐서플로우를 활용해 신경망을 학습시키는 코드를 다뤄본 적이 있다.
-   간단한 판별 모델링(분류, 회귀 등)의 개념을 알고, 실습해 본 적이 있다.

## 학습 목표

-   생성 모델링 개념을 이해하며 판별 모델링과의 차이 알기
-   Pix2Pix, CycleGAN 등의 이미지 관련 다양한 생성 모델링의 응용을 접하며 흥미 가지기
-   Fashion MNIST 데이터셋의 의미를 알기
-   생성적 적대 신경망(GAN)의 구조와 원리를 이해하기
-   텐서플로우로 짠 DCGAN 학습 코드를 익히며 응용하기
<br></br>

## 없던 데이터를 만들어낸다, 생성 모델링

앞서 진행했었던 이미지을 특정 카테고리로 분류하는 모델은 판별 모델링 (Discriminative Modeling) 이라고 한다. 즉, 데이터를 입력받아 특정 기준으로 판별하는 것이 목표인 것이다.

생성 모델링 (Generative Modeling) 이란 없던 데이터를 생성해내는 것이 목표이다. 즉, 학습한 데이터셋과 비슷하지만 기존에는 없는 새로운 데이터셋을 생성하는 모델이다.

응용하기에 따라 생성 모델을 통해 다른 판별 모델을 학습하는데 필요한 데이터를 만들어 내거나, 음악, 글, 이미지를 학습한 후 새로운 음악, 글, 이미지 등을 생성해낼 수 있다.

판별 모델링은 생성자 (Generator) 와 판별자 (Discriminator), 두 가지 네트워크가 사용되며, 생성자는 새로운 무엇가를 만들어내는 역할을 하며, 판별자는 생성자가 만들어낸 것을 해석하여 피드백을 제공하는 역할을 한다.

+ 참고 : [DeepComposer 모델 시연 영상](https://youtu.be/XH2EbK9dQlg)
<br></br>

## 여러 가지 생성 모델링 기법과 친해지기 (01). Pix2Pix

### 그림으로 사진을 변환해 보자 : Pix2Pix

Pix2Pix 는 간단한 이미지를 입력할 때, 실제 사진처럼 보이도록 바꿔줄 때 많이 사용되는 모델이다.
Input image 를 입력받아 내부 연산을 통해 실제 사진과 같은 형상으로 변환된 Predicted Image 를 출력하는 것이다.

학습이 진행되면서 모델이 생성한 Predicted Image 를 판별자가 평가하게 되며, 이에 따라 Predicted Image 와 Ground Truth 이미지와 많이 달랐다가 점차 실제 와 같은 결과물을 만들어내게된다.

이렇게 한 이미지를 다른 이미지로 픽셀 단위로 변환한다는 뜻의 Pixel to Pixel 을 본따 Pix2Pix 라는 이름이 붙었다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/pix2pix.max-800x600.png)
<br></br>
위 그림의 윗 부분과 같이 단순화된 이미지 (Input Image) 와 아래 부분과 같은 실제 이미지 (Ground Truth) 가 쌍을 이루는 데이터셋으로 학습을 진행한다.

Input Image 는 매우 단순화된 이미지이며, 대략적인 정보만 알 수 있을 뿐 구체적으로 무엇인지 분간하기 힘들다.

Predicted Image 는 Input Image 의 이런 대략적인 구조 정보만을 이용해 어울리는 세부 디자인을 생성하며, 결과물은 Ground Truth 와 완벽하게 똑같지는 않지만 전체적으로 비슷하게 생성이 가능하다.

이러한 Pix2Pix 를 통해 스케치로 그려진 그림을 실사화, 흑백 사진을 컬러로 변환, 위성 사진을 지도 이미지도 변환, 낯 배경을 밤 배경으로 변환 등에 활용할 수 있다.

+ 참고 : [논문 : Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf)
<br></br>
## 여러 가지 생성 모델링 기법과 친해지기 (02). CycleGAN

### 모네의 그림을 사진으로, 사진을 다시 모네의 그림으로 : CycleGAN

![](https://aiffelstaticprd.blob.core.windows.net/media/images/CycleGAN.max-800x600.png)
<br></br>
CycleGAN 은 Pix2Pix 에서 발전된 모델이다. 이름에서 유추할 수 있듯이 한 이미지와 다른 이미지를 번갈아가며 Cyclic 하게 변환시킬 수 있다.

Pix2Pix 는 한 방향으로의 변환만 가능했던거와 달리 양방향으로 이미지 변환이 가능하다. 즉, 실사 이미지를 그림 이미지로, 그림 이미지를 다시 실사 이미지로 변환이 가능한 것이다.

또한 CycleGAN 은 Pix2Pix 가 쌍으로 된 데이터를 필요로 했던거와는 달리 쌍으로 된 데이터가 필요없다.

위 그림와 같이 말을 얼룩말로, 얼룩말을 말로 서로 변환할 경우 쌍을 이루지 않더라도 그냥 얼룩말이 있는 사진과 말이 있는 사진 각각의 데이터셋만 있다면 학습이 가능하다.

모델이 스스로 얼룩말과 말 데이터에서 각각의 스타일을 학습해서 새로운 이미지에 학습한 스타일을 입힐 수 있도록 설계되었기 때문이다.

딥러닝에서 쌍으로 된 데이터 (Paired Data) 가 필요없다는 것은 엄청난 메리트이다. 이는 데이터를 구하기 훨씬 쉽고, 라벨을 붙이는 주석 (Annotation) 비용이 필요없다는 의미이기 때문이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/CycleGAN2.max-800x600.jpg)<br></br>
위 그림은 CycleGAN 을 통해 실제 사진을 모네 등의 화가가 그린 그림처럼 바꾸는 것이며, 화가의 그림을 사진처럼 바꾸는 반대의 경우도 가능하다.
<br></br>

## 여러 가지 생성 모델링 기법과 친해지기 (03). Neural Style Transfer

### 사진에 내가 원하는 스타일을 입혀보자 : Neural Style Transfer

Neural Style Transfer 는 이미지의 스타일을 변환시키는 기법이다. 전체 이미지의 구성을 유지하고자하는 Base Image 와 입히고 싶은 스타일이 담긴 Style Image 두 장을 활용하여 새로운 이미지를 만들어 내는 것이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/StyleTransfer.max-800x600.png)
<br></br>

위 그림에서 왼쪽 맨 위 이미지가 Base Image 이다. 나머지 이미지에 작제 붙어 있는 이미지가 Style Image 이다.

Base Image 를 제외하고 나머지 이미지를 결과를 보면, Base Image 의 특징인 건물, 강, 하늘을 유지하면서 Style Image 의 느낌을 잘 적용한 것을 확인할 수 있다.

즉, Base Image 에서는 Content (내용) 만, 그리고 Style Image 에서는 Style (스타일) 만 추출해서 합친 결과물이다.

최종 결과를 만드는 과정에는 Base Image 의 내용은 잃지 않으면서, Style Image 의 스타일을 효과적으로 입히기 위한 정교한 손실함수들을 통한 다양한 최적화가 포함되어 있다. 
<br></br>

## 패션을 디자인하려면? 먼저 패션을 배워야지! (01)

### Fashion MNIST

멋진 패션을 디자인하기 위해서는 다양한 패션을 경험해보면서 어떤것이 좋은 패션인지 학습해야한다.

이를 토대로 나만의 패션을 완성 해 나갈 수 있다. 생성 모델도 이와 같이 여러 패션을 접하면서 옷, 신발, 드레스와 같은 것이 어떻게 생겼는지 학습하고, 이를 토대로 새로운 디자인을 만들어내야 한다.

새로운 패션을 디자인하는 생성 모델을 구현하기 위해 Fashion MNIST 를 사용한다.

+ 참고 : [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
<br></br>

## 패션을 디자인하려면? 먼저 패션을 배워야지! (02). 코드로 살펴보기

생성 모델을 위해서는 텐서플로우 및 이미지와 GIF 파일을 다루는 imageio, display, matplotlib, PIL 등 다양한 패키지를 필요로 한다.
<br></br>
> imageio, pillow 라이브러리 설치
```python
pip install imageio 
pip install Pillow
```
<br></br>
> 작업 디렉토리 설정
```bash
$ mkdir -p ~/aiffel/dcgan_newimage/fashion/generated_samples

$ mkdir -p ~/aiffel/dcgan_newimage/fashion/training_checkpoints

$ mkdir -p ~/aiffel/dcgan_newimage/fashion/training_history
```
<br></br>
> 사용할 라이브러리 가져오기
```python
import os
import glob
import time

import PIL
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
%matplotlib inline

print("tensorflow", tf.__version__)
```
`fashion_mnist` 데이터는 우리가 인터넷에서 따로 다운받을 필요 없이,`tf.keras` 안에 있는 `datasets` 에 이미 들어가 있어서 꺼내기만 하면된다.
<br></br>
> fashion MNIST 데이터셋 가져오기
```python
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_x, _), (test_x, _) = fashion_mnist.load_data()
```
`load_data()`로 데이터를 가져올 수 있으며, 생성 모델에서는 분류 문제에서와 달리, 각 이미지가 어떤 카테고리인지 나타내주는 라벨이 필요 없다. 

즉, `y_train`, `y_test`에 해당하는 데이터를 쓰지 않는다. 때문에 코드에서 `_` (언더스코어)로 해당 데이터들은 무시한다.

+ 참고 : [파이썬 언더스코어(_)에 대하여](https://mingrammer.com/underscore-in-python/#2-%EA%B0%92%EC%9D%84-%EB%AC%B4%EC%8B%9C%ED%95%98%EA%B3%A0-%EC%8B%B6%EC%9D%80-%EA%B2%BD%EC%9A%B0)
<br></br>
> 이미지의 각 픽셀값 확인
```python
print("max pixel:", train_x.max())
print("min pixel:", train_x.min())
```
Fashion MNIST 는 28 x 28 픽셀 이미지로 각 픽셀은 0 ~ 255 사이의 정수값을 가진다.
<br></br>
> 각 필셀을 정규화
```python
train_x = (train_x - 127.5) / 127.5 # 이미지를 [-1, 1]로 정규화합니다.

print("max pixel:", train_x.max())
print("min pixel:", train_x.min())
```
각 픽셀을 -1, 1 로 정규화시켜서 사용할 예정이므로, 중간값을 0 으로 맞춰주기 위해 127.5 를 뺀 후 127.5 로 나눠준다.
<br></br>
> train 데이터셋의 shape 확인
```python
train_x.shape
```
train 데이터셋에는 6만 장의 이미지가 있으며, 이미지 사이즈가 28 x 28 임을 확인할 수 있다.
<br></br>

앞서 CNN (합성곱) 계층을 다룰 때 배웠듯, 딥러닝에서 이미지를 다루려면 채널 수 에 대한 차원이 필요하다.

입력되는 이미지 데이터의 채널 수는 어떤 이미지인지에 따라 달라진다. 컬러의 경우 3 개의 채널이이며, 흑백의 경우 1 개의 채널을 가진다.

해당 데이터셋의 이미지는 흑백이므로 흑백 이미지에 맞는 채널 수의 차원을 추가해 줘야한다.
<br></br>
> Faashion MNIST 의 채널 수에 대한 차원 추가
```python
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
train_x.shape
```
<br></br>
> 첫 번째 데이터를 꺼내 시각화를 통해 확인
```python
plt.imshow(train_x[0].reshape(28, 28), cmap='gray')
plt.colorbar()
plt.show()
```
첫 번째 이미지는 신발임을 확인할 수 있다.

`plt.colorbar()` 를 이용해 오른쪽에 각 픽셀의 값과 그에 따른 색을 확인해볼 수 있으며, 픽셀에는 우리가 정규화 해준 대로 -1 ~ 1 사이의 값을 가지고, -1 이 가장 어두운 검은색, 1이 가장 밝은 흰색을 띤다고 표시되어 있다.
<br></br>
> 10 개 정도의 이미지를 시각화를 통해 확인
```python
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_x[i].reshape(28, 28), cmap='gray')
    plt.title(f'index: {i}')
    plt.axis('off')
plt.show()
```
`plt.figure(figsize=(10, 5))` 는 이미지의 전체 프레임의 크기를 결정한다.

여러 개의 이미지를 한 번에 띄우고 싶을 때에는 `plt.subplot(row, col, index)`의 형태로 볼 수 있다. 예를들어 10 개의 이미지를 2 x 5의 배열 형태로 보고 싶은 경우, `plt.subplot(2, 5, index)`로 작성한다.

index 는 1 부터 10 까지 순서대로 바뀌어야 하니 for 문에서 사용하는 `i`에 `1`을 `i+1`을 넣어주었다.

추가적으로 `plt.title('title')` 함수를 이용해서 이미지에 제목으로 라벨값을 넣어줬고, `plt.axis('off')` 함수로 불필요한 축을 지워주었다.
<br></br>
> 모델 학습을 위한 미니배치 학습 수행
```python
BUFFER_SIZE = 60000
BATCH_SIZE = 256
```
미니배치 학습이란 너무 많은 양의 학습이 필요할 때 적절한 사이즈로 잘라서 학습을 수행하는 것을 의미한다.

`BUFFER_SIZE` 은 전체 데이터를 섞기 위해 60,000 으로 설정하며, `shuffle()` 함수는 데이터셋을 잘 섞어서 모델에 넣어주는 역할을 한다.

+ 참고 : [텐서플로우 공식 문서: tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)
<br></br>
> 텐서플로우의 `Dataset`을 이용해 모델 학습 준비
```python
train_dataset = tf.data.Dataset.from_tensor_slices(train_x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```
tf.data.Dataset 모듈의 `from_tensor_slices()` 함수를 사용하면 리스트, 넘파이, 또는 텐서플로우의 텐서 자료형에서 데이터셋을 만들 수 있다.

`train_x` 라는 넘파이 배열 자료를 섞고, 이를 배치 사이즈에 따라 나누도록하였으며 데이터가 잘 섞이게 하기 위해서는 버퍼 사이즈를 총 데이터 사이즈와 같거나 크게 설정하는 것이 좋다.
<br></br>

## 그림을 만들어내는 화가 생성자, 그리고 평가하는 비평가 판별자 (01). GAN 이해하기

### GAN 이란?

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GAN.max-800x600.png)
<br></br>
모델 학습을 위해 모델을 만들어보자. 생성 모델 중 가장 간단한 형태는 바로 GAN (Generative Adversarial Network) 이다. 

생성자 (Generator) 와 판별자 (Discriminator), 2 가지의 네트워크를 가지며 생성자는 아무 의미 없는 랜덤 노이즈로부터 신경망에서의 연산을 통해 이미지 형상의 벡터를 생성해내는, 즉 무에서 유를 창조하는 역할을 한다.

판별자는 기존에 있던 진짜 이미지와 생성자가 만들어낸 이미지를 입력으로 받아 각 이미지가 Real (진짜 이미지) 인지 Fake (생성자가 만든 이미지) 인지에 대한 판단 정도를 실수값으로 출력한다.

GAN 모델의 최종적인 목표는 생성자가 만들어낸 것을 판별자가 진짜인지 가짜인지 구분할 수 있을 때 까지 반복하여 개선하는 것을 목표로 한다.

이렇게 생성자와 판별자가 경쟁하듯 이루어진 모델 구조때문에 Adversarial (적대적인) 이라는 이름을 사용한 것이다.

+ 참고 : [GAN 에 관한 OPEN AI](https://openai.com/blog/generative-models/)
<br></br>

## 그림을 만들어내는 화가 생성자, 그리고 평가하는 비평가 판별자 (02). 생성자 구현하기

우리가 구현하고자 하는 GAN 모델은 DCGAN (Deep Convolution GAN) 이라는 GAN 의 개량된 모델이다.

Keras 의 Sequential API 를 활용하여 구현한다.

+ 참고 :  [러닝 텐서플로- Chap07.3 - 텐서플로 추상화와 간소화, Keras](https://excelsior-cjh.tistory.com/159)
<br></br>
> 생성자 모델 구현
```python
def make_generator_model():

    # Start
    model = tf.keras.Sequential()

    # First: Dense layer
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Second: Reshape layer
    model.add(layers.Reshape((7, 7, 256)))

    # Third: Conv2DTranspose layer
    model.add(layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Fourth: Conv2DTranspose layer
    model.add(layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Fifth: Conv2DTranspose layer
    model.add(layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, \
                                     activation='tanh'))

    return model
```
`make_generator_model` 이라는 함수를 만들어서 언제든 생성자를 생성할 수 있도록 하였으며, 함수 내부에서는 먼저 `tf.keras.Sequential()`로 모델을 시작한 후 레이어를 쌓아준다.

가장 중요한 레이어는 `Conv2DTranspose` 레이어이다.

`Conv2DTranspose` 레이어는 이미지 사이즈를 넓혀주는 층으로, 해당 모델에서 3 번의 `Conv2DTranspose` 층을 이용해 `(7, 7, 256) → (14, 14, 64) → (28, 28, 1)` 순으로 이미지를 키워나간다.

이때 최종사이즈는 데이터셋의 이미지의  shape 와 동일하다.

+ 참고 : [What is Transposed Convolutional Layer?](https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11)
<br></br>

레이어의 사이사이에 특정 층들이 반복되는 것을 확인할 수 있는데, `BatchNormalization` 레이어는 신경망의 가중치가 폭발하지 않도록 가중치 값을 정규화시켜준다.

또한 중간층들의 활성화 함수는 모두 `LeakyReLU` 를 사용하였으며, 마지막 층에는 활성화 함수로 `tanh` 를 사용하는데, 이는 우리가 -1 ~ 1 이내의 값으로 픽셀값을 정규화시켰던 데이터셋과 동일하게 하기 위함이다.

-   참고:  
	- [라온피플-Batch Normalization](https://m.blog.naver.com/laonple/220808903260)
	- [활성화 함수 (activation function)](https://newly0513.tistory.com/20)
<br></br>

정리하면 생성자의 입력 벡터는 (batch_size, 100) 형상 의 노이즈 벡터이며, 이렇게 입력된 벡터는 7 x 7 x 256  = 12544 개의 노드를 가진 Dense 레이어를 거쳐 (batch_size, 12544) 형상의 벡터가 된다.

이렇게 Dense 레이어를 지난 후 Reshape 레이어를지나면서 이후 레이어에서 Convolutional 연산을 할 수 있도록 1 차원 벡터를 (7, 7, 256) 형상의 3 차원 벡터로 변환한다.
<br></br>
> 생성자 모델을 `generator` 변수에 저장하고 모델의 세부내용을 출력
```python
generator = make_generator_model()

generator.summary()
```
<br></br>
> `shape = (1, 100)` 형상을 가지는 랜덤 노이즈 벡터를 생성
```python
noise = tf.random.normal([1, 100])
```
`tf.random.normal`을 이용하면 가우시안 분포에서 뽑아낸 랜덤 벡터로 이루어진 노이즈 벡터를 만들 수 있
<br></br>
> 위에서 생성한 랜덤 노이즈 벡터를 통해 생성자 모델 테스트
```python
generated_image = generator(noise, training=False)
generated_image.shape
```
텐서플로우 2.0 버전에서는 레이어와 모델에 call 메소드를 구현해 놓기 때문에, 방금 만들어진 생성자 모델에 입력값으로 노이즈를 넣고 바로 모델을 호출하면 간단히 결과 이미지가 생성된다.

다만 현재는 학습 중이 아니므로, `training=False`를 설정해 준다. 이는 Batch Normalization 레이어는 훈련 시기와 추론(infernce) 시기의 행동이 다르기 때문이다.
<br></br>
> 생성된 이미지 시각화
```python
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.colorbar()
plt.show()
```
`matplotlib` 라이브러리는 2 차원 이미지만 보여줄 수 있으므로 0 번째와 3 번째 축의 인덱스를 0 으로 설정해서 `(28, 28)` shape 의 이미지를 꺼낼 수 있도록 설정하였다.

결과로 -1 과 1 사이의 값에서 적당히 잘 생성된 것을 확인할 수 있으며, 현재는 학습 전 상태이기에 노이즈 같은 이미지가 생성되었다.
<br></br>

## 그림을 만들어내는 화가 생성자, 그리고 비평하는 비평가 판별자 (03). 판별자 구현하기

> 판별자 구현
```python
def make_discriminator_model():

    # Start
    model = tf.keras.Sequential()

    # First: Conv2D Layer
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Second: Conv2D Layer
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Third: Flatten Layer
    model.add(layers.Flatten())

    # Fourth: Dense Layer
    model.add(layers.Dense(1))

    return model
```
판별자는 가짜 이미지와 진짜 이미지를 입력으로 받아 진짜라도 판단하는 정도 값을 출력하므로, (28, 28, 1) 크기의 이미지가, 출력은 단 하나의 숫자 (진짜라고 판단하는 정도) 가 된다.

모델 구조를 살펴보면 `Conv2DTranspose` 층을 사용해서 이미지를 키워나갔던 생성자와 반대로, 판별자는 `Conv2D` 층으로 이미지의 크기를 점점 줄여나간다.

첫 번째 Conv2D 층에서 입력된 `[28, 28, 1]` 사이즈의 이미지는 다음 층을 거치며 `(28, 28, 1) → (14, 14, 64) → (7, 7, 128)`까지 줄어들며 마지막에는 `Flatten` 층을 사용해 3차원 이미지를 1차원으로 쭉 펴서 7 x 7 x 128 = 6272, 즉 (1, 6272) 형상의 벡터로 변환한다.

생성자의 `Reshape` 층에서 1 차원 벡터를 3 차원으로 변환했던 것과 정확히 반대 역할을 수행한다.

1 차원 벡터로 변환한 후에는 마지막 `Dense` 레이어를 거쳐 단 하나의 값을 출력한다.
<br></br>
> 판별자 모델은 `discriminator` 변수에 저장하고 모델의 세부내용 확인
```python
discriminator = make_discriminator_model()

discriminator.summary()
```
<br></br>
> 가짜 이미지를 판별자에 입력하여 모델 테스트
```python
decision = discriminator(generated_image, training=False)
decision
```
텐서 형태로 출력되며, 생성자 모델 테스트와 같이 학습 전이기에 큰 의미가 없는 값일 확률이 높다.
<br></br>

## 생성 모델이 똑똑해지기 위한 기나긴 여정 (01). 손실함수와 최적화 함수

딥모델 학습을 위해 꼭 필요한 2 가지가 있다. 바로 손실함수 (Loss Function) 와  최적화 함수 (Optimizer) 이다.
<br></br>

### 손실함수 (Loss Function)

GAN 은 기본적으로 손실함수를 교차 엔트로피 (Cross Entropy) 를 사용한다.

교차 엔트로피는 점점 가까워지길 원하는 두 값이 얼마나 큰 차이가 나는지를 정량적으로 계산하고자 할 때 사용한다.

판별자는 한 개의 이미지가 진짜인지 가짜인지를 나타내는 2 개의 클래스 간 분류 문제이므로, 이진 교차 엔트로피 (Binary Corss Entropy) 를 사용한다.

즉, 진짜 이미지에 대한 라벨을 1, 생성한 가짜 이미지에 대한 라벨을 0 으로 두었을 떄, 손실함수를 활용해 정량적으로 달성해야하는 목표는 다음과 같다.

생성자는 판별자가 Fake Image 에 대해 판별한 값, 즉 `D(fake_image)` 값이 `1` 에 가까워지는 것을 목표로 한다.

판별자는 Real Image 판별값, 즉 `D(real_image)` 는 `1` 에, Fake Image 판별값, 즉 `D(fake_image)` 는 `0` 에 가까워지는 것을 목표로 한다.

결국 손실함수에 들어가는 값은 판별자의 판별 값이 됨을 알 수 있다.

이제 손실함수를 설계 해 보자. 손실함수에 사용할 교차 엔트로피 함수는 `tf.keras.losses` 라이브러리 안에 있다.

교차 엔트로피를 계산하기 위해 입력할 값은 판별자가 판별한 값인데, 판별자 모델의 맨 마지막 Layer에는 값을 정규화시키는 sigmoid나 tanh 함수와 같은 활성화 함수가 없었기에 구분자가 출력하는 값은 범위가 정해지지 않아 모든 실숫값을 가지게 된다.

하지만 tf.keras.losses 의 BinaryCrossEntropy 클래스는 기본적으로 본인에게 들어오는 인풋값이 0 - 1 사이에 분포하는 확률값이라고 가정하기 떄문에 `from_logits`를 `True`로 설정해 주어야 `BinaryCrossEntropy`에 입력된 값을 함수 내부에서 sigmoid 함수를 사용해 0 ~ 1 사이의 값으로 정규화한 후 알맞게 계산할 수 있다.
<br></br>
> 교차엔트로피 손실함수 설계
```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```
<br></br>

`cross_entropy`를 활용해 계산할 loss들은 다음과 같은 `fake_output`, `real_output` 두 가지를 활용하게 된다.

-   `fake_output`  : 생성자가 생성한 Fake Image를 구분자에 입력시켜서 판별된 값, 즉  `D(fake_image)`

-   `real_output`  : 기존에 있던 Real Image를 구분자에 입력시켜서 판별된 값, 즉  `D(real_image)`

`ake_output` 과 `real_output` 을 각각 1 또는 0에 비교해야 하는데, `tf.ones_like()` 와 `tf.zeros_like()` 함수를 활용하여 비교한다.

이 함수들은 특정 벡터와 동일한 크기이면서 값은 1 또는 0 으로 가득 채워진 벡터를 만들고 싶을 때 사용한다.
<br></br>
> tf.ones_like()` 와 `tf.zeros_like()` 함수를 활용 실험
```python
vector = [[1, 2, 3],
          [4, 5, 6]]

tf.ones_like(vector)
```
`vector`와 형태는 같지만, 그 내용물은 모두 1인 벡터가 만들어 짐을 확인할 수 있다. 이를 통해 영벡터, 혹은 1 로 채워진 벡터를 만들 수 있다.
<br></br>
> `generator_loss` 를 구하는 함수 구현
```python
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```
`generator_loss` 는 `fake_output` 가 1 에 가까워지기를 바라므로, 다음과 같이 `tf.ones_like` 와의 교차 엔트로피값을 계산하면 된다.

즉, `cross_entropy(tf.ones_like(fake_output), fake_output)` 값은 fake_output 이 (Real Image 를 의미하는) 1에 가까울수록 작은 값을 가진다.
<br></br>
> `discriminator_loss` 를 산출하는 함수 구현
```python
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
```
`discriminator_loss` 는 `real_output` 값은 1 에 가까워지기를, `fake_output` 값은 0 에 가까워지기를 바라므로, 두 가지 loss 값을 모두 계산하며, `real_output` 은 1 로 채워진 벡터와, `fake_output` 은 0 으로 채워진 벡터와 비교하면 된다.

즉, 최종 `discriminator_loss` 값은 이 둘을 더한 값이다.
<br></br>
> 판별자가 real output, fake output 을 얼마나 정확히 판별하는지의 accuracy 를 계산하는 함수 구현
```python
def discriminator_accuracy(real_output, fake_output):
    real_accuracy = tf.reduce_mean(tf.cast(tf.math.greater_equal(real_output, tf.constant([0.5])), tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(tf.math.less(fake_output, tf.constant([0.5])), tf.float32))
    return real_accuracy, fake_accuracy
```
판별자가 Real Ouput, Fake Output 을 얼마나 정확히 판별하는지를 나타내는 accuracy 를 계산해보는 것은 매우 중요하며, 이 둘의 정확도를 따로 계산하여 비교해보는 것이 중요하다.

이 정확도가 1 에 가깝다면 판별자가 쉽게 판별해 내고 있다는 의미로 생성자가 판별자를 잘 속이고 있지 못하다는 의미가 된다. 따라서 최종 목표는 해당 적확도가 낮아지는 것이다.

위 함수에서 사용된 Tensorflow 함수들의 역할을 예를 들어 순차적으로 정리하면 다음과 같다.

-   (1) tf.math.greater_equal (real_output, tf.constant([0.5]) : real_output 의 각 원소가 0.5 이상인지 True, False로 판별
   -   `>> tf.Tensor([False, False, True, True])`

-   (2) tf.cast( (1), tf.float32) : (1) 의 결과가 True 이면 1.0, False 이면 0.0 으로 변환
    -   `>> tf.Tensor([0.0, 0.0, 1.0, 1.0])`

-   (3) tf.reduce_mean( (2)) : (2)의 결과를 평균내어 이번 배치의 정확도(accuracy) 를 계산
    -   `>> 0.5`
<br></br>

### 최적화 함수 (Optimizer)

최적화 함수로는 Adam 최적화 기법을 활용한다.

+ 참고 : [문과생도 이해하는 딥러닝 (8) - 신경망 학습 최적화](https://sacko.tistory.com/42)
<br></br>
Adam 함수 또한 `tf.keras.optimizers` 안에 있으며, 중요한 하이퍼 파라미터인 "learning rate" 는 0.0001 로 설정할 텐데, 학습 품질을 올려보고 싶다면 여러 가지로 값을 바꾸어 가며 학습을 진행해 보면 된다.

중요한 점은 생성자와 구분자는 따로따로 학습을 진행하는 개별 네트워크이기 때문에 optimizer를 따로 만들어주어야 한다는 것이다.
<br></br>
> 최적화 함수 설계
```python
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```
<br></br>
> 학습 진행과정을 확인해보기 위해 매 학습마다 생성자가 생성한 샘플을 확인하도록 설정
```python
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])
seed.shape
```
샘플은 한 번에 16 장을 생성하도록 설정했으며, 생성할 샘플은 매번 같은 노이즈로 생성해야 그에 대한 진전 과정을 확인할 수 있으므로, 고정된 seed 노이즈를 만들어둬야 한다.

즉, 100차원의 노이즈를 총 16개, `(16, 100)` 형상의 벡터를 만들어 두도록 설정하였다.
<br></br>

## 생성 모델이 똑똑해지기 위한 기나긴 여정 (02). 훈련과정 설계

훈련 시 미니배치 당 진행할 `train_step` 함수를 먼저 만들어야하며, 이를 위해 학습시킬 훈련 함수 위에 `@tf.function` 이라는 데코레이터를 붙여서 사용한다.

+ 참고 : 
	+ [python decorator (데코레이터) 어렵지 않아요](https://bluese05.tistory.com/30)
	+ [Tensorflow Tutorial](https://www.tensorflow.org/api_docs/python/tf/function)
<br></br>
> 데코레이터를 코드로 이해해보기
```python
import numpy as np
import tensorflow as tf

def f(x, y):
  print(type(x))
  print(type(y))
  return x ** 2 + y

x = np.array([2, 3])
y = np.array([3, -2])
f(x, y)
```
<br></br>
```python
import numpy as np
import tensorflow as tf

@tf.function    # 위와 동일한 함수이지만 @tf.function 데코레이터가 적용되었습니다.
def f(x, y):
  print(type(x))
  print(type(y))
  return x ** 2 + y

x = np.array([2, 3])
y = np.array([3, -2])
f(x, y)
```
넘파이 배열을 입력으로 x, y 를 동일하게 사용했지만 f(x,y)의 결과 타입은 다르다. 

`@tf.function` 데코레이터가 사용된 함수에 입력된 입력은 Tensorflow 의 graph 노드가 될 수 있는 타입으로 자동변환된다.
<br></br>
> 훈련 시 미니배치 당 진행할 `train_step` 함수 생성
```python
@tf.function
def train_step(images):  #(1) 입력데이터
    noise = tf.random.normal([BATCH_SIZE, noise_dim])  #(2) 생성자 입력 노이즈

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  #(3) tf.GradientTape() 오픈
        generated_images = generator(noise, training=True)  #(4) generated_images 생성

        #(5) discriminator 판별
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        #(6) loss 계산
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        #(7) accuracy 계산
        real_accuracy, fake_accuracy = discriminator_accuracy(real_output, fake_output) 
    
    #(8) gradient 계산
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    #(9) 모델 학습
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, real_accuracy, fake_accuracy  #(10) 리턴값
```
`train_step` 함수를 하나하나 뜯어보면 다음과 같다.

-   (1) 입력데이터: Real Image 역할을 할  `images`  한 세트를 입력으로 받음

-   (2) 생성자 입력 노이즈 : generator 가 FAKE IMAGE 를 생성하기 위한  `noise` 를  `images`  한 세트와 같은 크기인  `BATCH_SIZE`  만큼 생성함

-   (3)  `tf.GradientTape()` 는 가중치 갱신을 위한 Gradient 를 자동 미분으로 계산하기 위해  `with`  구문 열기

-   (4) generated_images 생성 : generator가  `noise`를 입력받은 후  `generated_images`  생성

-   (5) discriminator 판별 : discriminator 가 Real Image 인  `images`와 Fake Image 인  `generated_images` 를 각각 입력받은 후  `real_output`,  `fake_output`  출력

-   (6) loss 계산 :  `fake_output`,  `real_output`으로 generator 와 discriminator 각각의 loss 계산

-   (7) accuracy 계산 :  `fake_output`,  `real_output` 으로 discriminator 가 얼마나 쉽게 판별하는지 정도를 계산

-   (8) gradient 계산 :  `gen_tape` 와  `disc_tape` 를 활용해 gradient 를 자동으로 계산

-   (9) 모델 학습 : 계산된 gradient 를 optimizer 에 입력해 가중치 갱신

-   (10) 리턴값 : 이번 스텝에 계산된 loss 와 accuracy 를 리턴

위 과정을 통해 1 번의 `train_step` 이 끝난다.
<br></br>
> 한 단계씩 학습할 train_step 과 함께 일정 간격으로 학습 현황을 볼 수 있는 샘플을 생성하는 함수 생성
```python
def generate_and_save_images(model, epoch, it, sample_seeds):

    predictions = model(sample_seeds, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('{}/dcgan_newimage/fashion/generated_samples/sample_epoch_{:04d}_iter_{:03d}.png'
                    .format(os.getenv('HOME'), epoch, it))

    plt.show()
```
위에서 구현한 고정된 seed 에 대한 결과물이 얼마나 나아지고 있는지를 확인할 수 있다.

model 이 16 개의 seed 가 들어있는 `sample_seeds` 를 입력받아서 만들어낸 `prediction` 을 시각화해주며, `plt` 에 저장되어 보여지는 이미지를 `plt.savefig` 로 간단히 파일화 해서 저장한다.

`generated_samples` 라는 폴더 아래에 저장해야 하므로 `try` 구문을 이용해 폴더가 없는 상황에 에러가 발생하는 것을 방지하였다. 만약 폴더가 없어서 에러가 난다면, except 구문으로 들어가 `os.mkdir('./generated_samples')`로 폴더를 만든 후 파일을 저장할 수 있다.
<br></br>
> 학습 과정 시각화
> (Loss, Accuracy 과정 시각화)
```python
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6    # matlab 차트의 기본 크기를 15,6으로 지정해 줍니다.

def draw_train_history(history, epoch):
    # summarize history for loss  
    plt.subplot(211)  
    plt.plot(history['gen_loss'])  
    plt.plot(history['disc_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('batch iters')  
    plt.legend(['gen_loss', 'disc_loss'], loc='upper left')  

    # summarize history for accuracy  
    plt.subplot(212)  
    plt.plot(history['fake_accuracy'])  
    plt.plot(history['real_accuracy'])  
    plt.title('discriminator accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('batch iters')  
    plt.legend(['fake_accuracy', 'real_accuracy'], loc='upper left')  
    
    # training_history 디렉토리에 epoch별로 그래프를 이미지 파일로 저장합니다.
    plt.savefig('{}/dcgan_newimage/fashion/training_history/train_history_{:04d}.png'
                    .format(os.getenv('HOME'), epoch))
    plt.show()
```
`train_step()` 함수가 리턴하는 `gen_loss`, `disc_loss`, `real_accuracy`, `fake_accuracy` 이상 4 가지 값을 history 라는 딕셔너리구조에 리스트로 저장하고 있다가 매 epoch 마다 시각화하는 함수를 구현하였으며, list 로 접근할 수 있도록 관리한다.
<br></br>
> 정기적으로 모델을 저장하기 위한 checkpoint 생성
```python
checkpoint_dir = os.getenv('HOME')+'/dcgan_newimage/fashion/training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
```
`tf.train.Checkpoint` 를 활용하면 매번 모델을 직접 저장해주지 않아도, 코드 한 줄로 빠르고 편하게 버전 관리를 할 수 있다.

모델이 복잡해지고 학습 속도가 오래 걸릴수록, 모델에 대한 저장 및 버전 관리는 필수적이며, checkpoint 에는 optimizer 와 생성자, 구분자를 모두 넣어 저장한다. 이는 생성자와 구분자가 학습한 모델 가중치를 저장하는 것이다.

checkpoint 모델을 저장하기 위해 작업환경 내에 `training_checkpoints` 라는 디렉토리를 사용하였다.
<br></br>

## 생성 모델이 똑똑해지기 위한 기나긴 여정 (03). 학습 시키기

> 모델 학습을 위해 합쳐주기
```python
def train(dataset, epochs, save_every):
    start = time.time()
    history = {'gen_loss':[], 'disc_loss':[], 'real_accuracy':[], 'fake_accuracy':[]}

    for epoch in range(epochs):
        epoch_start = time.time()
        for it, image_batch in enumerate(dataset):
            gen_loss, disc_loss, real_accuracy, fake_accuracy = train_step(image_batch)
            history['gen_loss'].append(gen_loss)
            history['disc_loss'].append(disc_loss)
            history['real_accuracy'].append(real_accuracy)
            history['fake_accuracy'].append(fake_accuracy)

            if it % 50 == 0:
                display.clear_output(wait=True)
                generate_and_save_images(generator, epoch+1, it+1, seed)
                print('Epoch {} | iter {}'.format(epoch+1, it+1))
                print('Time for epoch {} : {} sec'.format(epoch+1, int(time.time()-epoch_start)))

        if (epoch + 1) % save_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epochs, it, seed)
        print('Time for training : {} sec'.format(int(time.time()-start)))

        draw_train_history(history, epoch)
```
앞서 구현한 한 단계를 학습하는 `train_step`, 샘플 이미지를 생성하고 저장하기 위한 `generate_and_save_images()`, 학습 과정을 시각화하는 `draw_train_history()`, 그리고 모델까지 저장하기 위한 `checkpoint` 를 한 곳에 합쳐주면 된다.
<br></br>
> 하이퍼 파마리터를 설정한 후 훈련
```python
save_every = 5
EPOCHS = 50

# 사용가능한 GPU 디바이스 확인
tf.config.list_physical_devices("GPU")

%%time
train(train_dataset, EPOCHS, save_every)

# 학습과정의 loss, accuracy 그래프 이미지 파일이 ~/aiffel/dcgan_newimage/fashion/training_history 경로에 생성되고 있으니
# 진행 과정을 수시로 확인해 보시길 권합니다.
```
`train()` 함수를 실행하여 모델이 학습하면서 만들어내는 결과물을 실시간으로 확인할 수 있다.
<br></br>
> 학습과정 시각화
```python
anim_file = os.getenv('HOME')+'/dcgan_newimage/fashion/fashion_mnist_dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('{}/dcgan_newimage/fashion/generated_samples/sample*.png'.format(os.getenv('HOME')))
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

!ls -l ~/dcgan_newimage/fashion/fashion_mnist_dcgan.gif
```
학습이 끝난 후 우리가 생성했던 샘플 이미지들을 합쳐 GIF 파일로 만들어 주어 학습 진행 과정을 확인해 볼 수 있다.

GIF 파일은 우리가 오래전에 import 해놓았던 `imageio` 라이브러리를 활용해 만들 수 있으며, `imageio.get_writer` 를 활용해서 파일을 열고, 거기에 `append_data` 로 이미지를 하나씩 붙여나가는 방식이다.

`fasion_mnist_dcgan.gif` 파일이 저장된다.
<br></br>
