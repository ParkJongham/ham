# 20. 난 스케치를 할 테니 너는 채색을 하거라


## 학습 전제

1.  신경망의 학습 방법에 대한 전반적인 절차를 알고 있어야 합니다.
2.  CNN, GAN에 대한 기본적인 개념을 알고 있어야 합니다.
3.  Tensorflow의 Subclassing API로 레이어 및 모델을 구성하는 방법에 대해 대략적으로 알고 있어야 합니다.
4.  Tensorflow의 GradientTape API를 이용한 학습 코드를 보고 이해할 수 있어야 합니다.
5.  (중요) Tensorflow 내에서 잘 모르는 함수(또는 메서드)를 발견했을 때, 공식 문서에서 검색하고 이해해보려는 의지가 필요합니다.


## 학습 목표

1.  조건을 부여하여 생성 모델을 다루는 방법에 대해 이해합니다.
2.  cGAN 및 Pix2Pix의 구조와 학습 방법을 이해하고 잘 활용합니다.
3.  CNN 기반의 모델을 구현하는데 자신감을 갖습니다.
<br></br>

## 조건 없는 생성모델 (Unconditional Generative Model), GAN

![](https://aiffelstaticprd.blob.core.windows.net/media/images/mnist_results.max-800x600.png)
<br></br>

위 그림은 GAN 을 통해 MNIST 데이터셋을 학습하여 생성한 것이다.
GAN 모델이 학습을 성공적으로 마쳤다면 위 그림과 같이 실제 손글씨와 유사한 손글씨 이미지를 생성해낸다.

이때  GAN 모델은 어떻게 이미지를 출력할까? 아마도 다양한 노이즈를 계속 입력으로 넣어보고 특정 이미지가 생성되기를 기다릴 것이다. 즉, 한번에 생성될 수도 있지만 몇 번의 시도 끝에 성공할 수 있을지는 아무도 알 수 없다.

이렇게 학습된 GAN 을 이용하 실제 이미지를 생성할 때 아쉬운 점은 원하는 종류의 이미지를 즉각적으로 생성할 수 없다는 것이다. 즉, 일반적인 GAN 과 같은 unconditioned generative model 은 내가 생성하고자 하는 데이터에 대해 제어하기 어렵다.
<br></br>

## 조건 있는 생성모델 (Conditional Generative Model), cGAN

cGAN 은 Conditional Generative Adversarial Nets 의 약자로, 원하는 종류의 이미지를 생성하고자 할 때, GAN 이 가진 생성 과정의 아쉬운부분인 생성하고자 하는 데이터에 제어가 어려운 점을 개선할 수 있도록 고안된 방법다.
<br></br>

### GAN 의 목적함수

GAN 은 Generator 와 Discriminator 라 불리는 생성자와 판별자에 해당하는 두 신경망이 minimax game 을 통해 서로 경쟁하며 발전한다.

이를 수식으로 표현하면 $$\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}~(x)}[log D(x)] + \mathbb{E}_{z\sim p_x(z)}[log(1-D(G(z)))]$$

와 같으며 생성자는 이 식의 결과를 최소화하려하며, 판별자는 이 식의 결과를 최대화하려 학습한다.

위 식에서 $z$ 는 임의 노이즈를, $D$ 와 $G$ 는 각각 Discriminator 및 Generator 를 의미한다.

먼저 판별자, $D$ 의 입장에서 수식을 생각할 때, 실제 이미지를 1, 가짜 이미지를 0 으로 두었을 때 $D$ 는 식을 최대화해야한다.

따라서 우변의 + 를 기준으로 양쪽의 항 $(logD(x))$ 및 $log(1−D(G(z)))$ 모두 최대가 되게 해야하기 때문에 두개의 log 가 1 이 되게 해야한다.

즉,  $D(x)$ 는 1 이 되도록, $D(G(z))$ 는 0 이 되도록 해야하며, 이는 진짜 데이터 $(x)$ 를 진짜로, 가짜 데이터 $(G(z))$ 를 가짜로 정확히 예측하도록 학습한다는 의미이다.

이제 생성자 $G$ 의 입장에서 수식을 생각해 보면, $D$ 와 반대로 $G$ 는 위 식을 최소화해야 하고 위 수식에서는 우변의 첫 번째 항은 $G$ 와 관련이 없으며, 마지막 항 $log(1−D(G(z))$ 만을 최소화해야 한다. 

$log(1−D(G(z))$ 을 최소화 한다는 것은 log 내부가 0 이 되도록 해야하며, 이는 $D(G(z))$ 가 1 이 되로록 하는 것과 같다.

즉, $G$ 는 $z$ 를 입력받아 생성한 데이터 $G(z)$ 를 $D$ 가 진짜 데이터라고 예측할 만큼 진짜 같은 가짜 데이터를 만들도록 학습한다.

<br></br>
### cGAN 의 목적함수

cGAN 의 목적함수는 다음과 같은 수식으로 표현한다.

$$\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}~(x)}[log D(x\lvert{y})] + \mathbb{E}_{z\sim p_x(z)}[log(1-D(G(z\lvert{y})))]$$

cGAN 의 목적함수는 GAN 의 목적함수와 비교하였을 때 $D(x)$ 와 $G(z)$ 가 각각 $D(x∣y), G(z∣y)$ 로 바뀜을 알 수 있다.

위 cGAN 의 목적함수 수식은 GAN 의 목적함수와 비교했을 때  우변의 + 를 기준으로 양쪽 항에 $y$ 가 추가된 것 뿐이다. 즉,  $
G$ 와 $D$ 의 입력에 특정 조건을 나타내는 정보인 $y$ 를 같이 입력한다는 것이다.

이외의 부분은 GAN 의 목적함수와 동일하게 각각 $y$ 를 추가로 입력받아 $G$ 의 입장에서 식을 최소화하고, $D$ 의 입장에서 식을 최대화하도록 학습한다.

이때 $y$ 는 어떤 정보여도 상관없다. 예를 들어 MNIST 데이터셋을 학습시키는 경우 $y$ 는 0 ~ 9 까지의 label 정보가 된다.

이를 통해 생성자가 노이즈 $z$ 를 입력받았을 때, 특정 조건 $y$ 를 함께 입력받아 $z$ 를 어떤 이미지로 만들어야하는지에 대한 방향을 제어하게된다.

<br></br>
### 그림으로 이해하기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/gan_img.max-800x600.png)
<br></br>

위 그림은 GAN 의 Feed Forward 과정을 나타낸 그림이다.

GAN 의 학습 과정에서 생성자는 노이즈 $z (파란색)$ 이 입력되고, 특정 representation (검정색) 으로 변환 후 가짜 데이터 $G(z)$  그림에서 빨간색을 생성한다.  

판별자는 실제 데이터 $x$ 와 생성자가 생성한 가짜 데이터 $G(z)$ 를 각각 입력으로 받아 $D(x)$ 와 $D(G(z))$, 그림에서 보라색을 계산하여 진짜와 가짜를 식별한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/cgan_img.max-800x600.png)
<br></br>
위 그림은 cGAN 의 학습 과정을 나타낸 그림으로 GAN  목적함수에서 $y$ 정보가 함께 입력된다.

여기서 생성자는 노이즈 $z (파란색)$ 와 추가 정보 $y (녹색)$ 함께 입력받아 생성자 내부에서 결합되어 representation (검정색) 으로 변환된다. 이 representation 은 가짜 데이터 $G(z∣y)$ 를 생성하며, MNIST 나 CIFAR - 10 등의 데이터셋에 대해 학습시킬 경우  $y$ 는 레이블 정보이며, 일반적으로 one - hot 벡터를 입력으로 넣는다.

판별자는 실제 데이터 $x$ 와  생성자가 생성한 가짜 데이터 $G(z∣y)$ 와 $y$ 정보를 각각 입력으로 받아 진짜와 가짜를 식별한다. MNIST 나 CIFAR-10 등의 데이터셋에 대해 학습시키는 경우 실제 데이터 $x$ 와 $y$ 는 알맞은 한 쌍 ("7"이라 쓰인 이미지의 경우 레이블도 7)을 이뤄야 하며, 마찬가지로 생성자에 입력된 $y$ 와 판별자에 입력되는 $y$는 동일한 레이블을 나타내야 한다.
<br></br>

## 내가 원하는 숫자 이미지 만들기 (01). Generator 구성하기

GAN 과 cGAN 을 구현해보자. MNIST 데이터셋을 이용하며, 코드는 [TF2-GAN](https://github.com/thisisiron/TF2-GAN) 를 참고하였다.

<br></br>
### 데이터 준비하기

> tensorflow-datasets 라이브러리 설치
```python
pip install tensorflow-datasets
```
<br></br>
> tensorflow-datasets 에서 MNIST 데이터셋을 불러와 확인
```python
import tensorflow_datasets as tfds

mnist, info =  tfds.load(
    "mnist", split="train", with_info=True
)

fig = tfds.show_examples(mnist, info)
```
여러개의 손글씨 숫자 이미지와 이에 맞는 레이블리 출력된다.
<br></br>
> 데이터 전처리
```python
import tensorflow as tf

BATCH_SIZE = 128

def gan_preprocessing(data):
    image = data["image"]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def cgan_preprocessing(data):
    image = data["image"]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    
    label = tf.one_hot(data["label"], 10)
    return image, label

gan_datasets = mnist.map(gan_preprocessing).shuffle(1000).batch(BATCH_SIZE)
cgan_datasets = mnist.map(cgan_preprocessing).shuffle(100).batch(BATCH_SIZE)
```
이미지 픽셀값을 -1 ~ 1 사이의 범위로 변경했고, 레이블 정보를 원-핫 인코딩 (one-hot encoding) 해준다.

GAN과 cGAN 각각을 실험해 보기 위해 label 정보 사용 유무에 따라 `gan_preprocessing()`과 `cgan_preprocessing()` 두 가지 함수를 생성하였다.
<br></br>
> 한 개 데이터셋을 선택해 전처리 함수가 잘 되는지 확인
```python
import matplotlib.pyplot as plt

for i,j in cgan_datasets : break

# 이미지 i와 라벨 j가 일치하는지 확인해 봅니다.     
print("Label :", j[0])
print("Image Min/Max :", i.numpy().min(), i.numpy().max())
plt.imshow(i.numpy()[0,...,0], plt.cm.gray)
```
이미지에 쓰인 숫자와 레이블이 일치해야 하고, 이미지 값의 범위가 -1 ~ 1 사이에 있어야 한다.

원 - 핫 인코딩으로 표현된 라벨과 출력 이미지는 0 과 1 로 이루어진 원-핫 벡터에는 각자 고유의 인덱스가 있기 때문에 MNIST 의 경우, 숫자 0 은 `[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]`, 숫자 6 은 `[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]`의 값을 가진다. 

를 통해 일치하는지 확인할 수 있다.
<br></br>

### GAN Generator 구성하기

Tensorflow2 의 Subclassing 방법을 이용하여 GAN Generator 를 구현해보자.

> GAN 의 Generator 구현
```python
from tensorflow.keras import layers, Input, Model

class GeneratorGAN(Model):
    def __init__(self):
        super(GeneratorGAN, self).__init__()

        self.dense_1 = layers.Dense(128, activation='relu')
        self.dense_2 = layers.Dense(256, activation='relu')
        self.dense_3 = layers.Dense(512, activation='relu')
        self.dense_4 = layers.Dense(28*28*1, activation='tanh')

        self.reshape = layers.Reshape((28, 28, 1))

    def call(self, noise):
        out = self.dense_1(noise)
        out = self.dense_2(out)
        out = self.dense_3(out)
        out = self.dense_4(out)
        return self.reshape(out)
```
Subclassing 방법은 tensorflow.keras.Model 을 상속받아 클래스를 만들며, 일반적으로 `__init__()` 메서드 안에서 레이어 구성을 정의하고, 구성된 레이어를 `call()` 메서드에서 사용해 forward propagation 을 진행한다.

Subclassing 방법은 Pytorch 의 모델 구성 방법과도 매우 유사하므로 이에 익숙해진다면 Pytorch 의 모델 구성 방법도 빠르게 습득할 수 있다.

`__init__()` 메서드 안에서 사용할 모든 레이어를 정의했으며, 4 개의 fully-connected 레이어 중 한 개를 제외하고 모두 ReLU 활성화를 사용한다.

`call()` 메서드에서는 노이즈를 입력받아 `__init__()`에서 정의된 레이어들을 순서대로 통과해  Generator 는 숫자가 쓰인 이미지를 출력해야 하므로 마지막 출력은 layers.Reshape() 을 이용해 (28, 28, 1) 크기로 변환된다.
<br></br>

### cGAN Generator 구성하기

> cGAN Generator 구현
```python
class GeneratorCGAN(Model):
    def __init__(self):
        super(GeneratorCGAN, self).__init__()
        
        self.dense_z = layers.Dense(256, activation='relu')
        self.dense_y = layers.Dense(256, activation='relu')
        self.combined_dense = layers.Dense(512, activation='relu')
        self.final_dense = layers.Dense(28 * 28 * 1, activation='tanh')
        self.reshape = layers.Reshape((28, 28, 1))

    def call(self, noise, label):
        noise = self.dense_z(noise)
        label = self.dense_y(label)
        out = self.combined_dense(tf.concat([noise, label], axis=-1))
        out = self.final_dense(out)
        return self.reshape(out)
```
GAN 의  Generator 와 차이점은 레이블 정보가 추가된다는 것 뿐이며, cGAN의 입력은 노이즈 및 레이블 정보, 2개이다.

`__init__()` 메서드에서 노이즈 및 레이블 입력 각각에 적용할 레이어를 생성했으며, (dense_z, dense_y)
 에 해당하는 노이즈 데이터를 입력은 각각 1 개의 fully-connected 레이어와 ReLU 활성화를 통과합니다. 

이후 (dense_z, dense_y) 의 결과가 (tf.concat, conbined_dense) 에서 서로 연결되이 다시 한번 1개의 fully-connected 레이어와 ReLU 활성화를 통과한다.

마지막으로 (tf.concat, conbined_dense) 의 결과가  (final_dense, reshape) 에서 1개의 fully-connected 레이어 및 Hyperbolic tangent 활성화를 거쳐 28 x 28 차원의 결과가 생성되고 (28, 28, 1) 크기의 이미지 형태로 변환되어 출력된다.
<br></br>

## 내가 원하는 숫자 이미지 만들기 (02). Discirminator 구성하기

### GAN Discirminator 구성하기

> GAN 의 Discriminator 구현
```python
class DiscriminatorGAN(Model):
    def __init__(self):
        super(DiscriminatorGAN, self).__init__()
        self.flatten = layers.Flatten()
        
        self.blocks = []
        for f in [512, 256, 128, 1]:
            self.blocks.append(
                layers.Dense(f, activation=None if f==1 else "relu")
            )
        
    def call(self, x):
        x = self.flatten(x)
        for block in self.blocks:
            x = block(x)
        return x
```
`__init__()`에 `blocks`라는 리스트를 하나 만들어 놓고, for loop를 이용하여 필요한 레이어들을 쌓는다.

이와 같은 방식을 통해 각각의 fully-connected 레이어를 매번 정의하지 않아 많은 레이어를 편하게 사용할 수 있다.

Discriminator 의 입력은 Generator 가 생성한 (28, 28, 1) 크기의 이미지이며, 이를 fully-connected 레이어로 학습하기 위해 `call()` 에서는 가장 먼저 `layers.Flatten()` 이 적용된다.

이후에 레이어들이 쌓여있는 `blocks`에 대해 for loop를 이용하여 레이어들을 순서대로 하나씩 꺼내 입력 데이터를 통과시키고 마지막 fully-connected 레이어를 통과하면 진짜 및 가짜 이미지를 나타내는 1 개의 값이 출력된다.
<br></br>

### cGAN Discirminator 구성하기

cGAN 의 Discirminator 에는 `Maxout`이라는 특별한 레이어가 사용된다.

`Maxout`은 간단히 설명하면 두 레이어 사이를 연결할 때, 여러 개의 fully-connected 레이어를 통과시켜 그 중 가장 큰 값을 가져오는 역할을 하며, 예를 들어 2개의 fully-connected 레이어를 사용할 때 `Maxout`을 식으로 표현하면 다음과 같다.

$$max(w_1^Tx+b_1,\ w_2^Tx+b_2)$$

+ 참고 : 
	- [[라온피플] Stochastic Pooling & Maxout](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=221259325819&proxyReferer=&proxyReferer=https:%2F%2Fwww.google.com%2F)
	- [[Paper] Maxout Networks](https://arxiv.org/pdf/1302.4389.pdf)
<br></br>
> `Maxout` 구현
```python
class Maxout(layers.Layer):
    def __init__(self, units, pieces):
        super(Maxout, self).__init__()
        self.dense = layers.Dense(units*pieces, activation="relu")
        self.dropout = layers.Dropout(.5)    
        self.reshape = layers.Reshape((-1, pieces, units))
    
    def call(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.reshape(x)
        return tf.math.reduce_max(x, axis=1)
```
`tensorflow.keras.layers.Layer` 를 상속받아 레이어를 정의하고, `__init__()`, `call()` 메서드를 구성한다.

`Maxout` 레이어를 구성할 때 `units` 과 `pieces` 의 설정이 필요하며, `units` 차원 수를 가진 fully-connected 레이어를 `pieces`개 만큼 만들고 그 중 최대 값을 출력한다.

예를 들면 사용할 `Maxout` 레이어가 `units=100`, `pieces=10`으로 설정 된다면 입력으로 부터 100차원의 representation 을 10 개 만들고, 10 개 중에서 최대값을 가져와 최종 1 개의 100 차원 representation 이 출력된다.

이를 수식으로 표현하면 다음과 같다.

$$max(w_1^Tx+b_1,\ w_2^Tx+b_2, \ …, \ w_9^Tx+b_9,\ w_{10}^Tx+b_{10})$$

위 코드에서는 각각 $wx + b$ 가 모두 100 차원이다.
<br></br>
> cGAN 의 Discriminator 구현
```python
class DiscriminatorCGAN(Model):
    def __init__(self):
        super(DiscriminatorCGAN, self).__init__()
        self.flatten = layers.Flatten()
        
        self.image_block = Maxout(240, 5)
        self.label_block = Maxout(50, 5)
        self.combine_block = Maxout(240, 4)
        
        self.dense = layers.Dense(1, activation=None)
    
    def call(self, image, label):
        image = self.flatten(image)
        image = self.image_block(image)
        label = self.label_block(label)
        x = layers.Concatenate()([image, label])
        x = self.combine_block(x)
        return self.dense(x)
```
위에서 정의한 `Maxout` 레이어를 3 번 사용하면 cGAN 의 Discriminator 를 구성할 수 있다.

GAN 의 Discriminator 와 마찬가지로 Generator 가 생성한 (28, 28, 1) 크기의 이미지가 입력되므로, `layers.Flatten()`이 적용된다. 그리고 이미지 입력 및 레이블 입력 각각은 `Maxout` 레이어를 한번씩 통과한 후 서로 결합되어 `Maxout` 레이어를 한번 더 통과하며 마지막 fully-connected 레이어를 통과하면 진짜 및 가짜 이미지를 나타내는 1 개의 값이 출력된다,

즉, cGAN의 Disciminator 의 연산 순서는 먼저 이미지가 Maxout 레이어를 통과하고, 레이블이 Maxout 레이어를 통과하고, 앞의 두 결과로 나온 representation 을 결합한 후 Maxout 레이어를 통과한다.

이때 (28, 28, 1) 크기 이미지 및 (10, ) 크기 레이블이 입력된다면 이미지가 Maxout 레이러를 통과하면 240 차원 수를 가지며, 레이블이 Maxout 레이어를 통과하면 50 차원 수를 가진다. 마지막으로 앞서 나온 각각의 결과 representation 를 결합후 Maxout 레이어를 통과하면 240 차원 수를 가지게 된다.
<br></br>

## 내가 원하는 숫자 이미지 만들기 (03). 학습 및 테스트하기

> GAN, cGAN 각각의 모델 학습에 공통적으로 필요한 Loss function, optimizer 정의
```python
from tensorflow.keras import optimizers, losses

bce = losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    return bce(tf.ones_like(real_output), real_output) + bce(tf.zeros_like(fake_output), fake_output)

gene_opt = optimizers.Adam(1e-4)
disc_opt = optimizers.Adam(1e-4)    
```
Generator 및 Discriminator 를 이용해 MINST 를 학습하고 각 모델로 직접 숫자 손글씨를 생성하기에 앞서 공통적으로 필요한 loss function 과 optimizer 를 정의한다.

진짜 및 가짜를 구별하기 위해 `Binary Cross Entropy`를 사용하고, `Adam optimizer`를 이용해 학습한다.
<br></br>

### GAN 으로 MNIST 학습하기

> 하나의 배치 크기 데이터로 모델을 업데이트하는 함수 생성
```python
gan_generator = GeneratorGAN()
gan_discriminator = DiscriminatorGAN()

@tf.function()
def gan_step(real_images):
    noise = tf.random.normal([real_images.shape[0], 100])
    
    with tf.GradientTape(persistent=True) as tape:
        # Generator를 이용해 가짜 이미지 생성
        fake_images = gan_generator(noise)
        # Discriminator를 이용해 진짜 및 가짜이미지를 각각 판별
        real_out = gan_discriminator(real_images)
        fake_out = gan_discriminator(fake_images)
        # 각 손실(loss)을 계산
        gene_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)
    # gradient 계산
    gene_grad = tape.gradient(gene_loss, gan_generator.trainable_variables)
    disc_grad = tape.gradient(disc_loss, gan_discriminator.trainable_variables)
    # 모델 학습
    gene_opt.apply_gradients(zip(gene_grad, gan_generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, gan_discriminator.trainable_variables))
    return gene_loss, disc_loss
```
앞서 구성한 GeneratorGAN 및 DiscriminatorGAN 모델 클래스를 이용하며, 입력으로 사용되는 노이즈를 100차원으로 설정했으며, 하나의 배치 크기 데이터로 모델을 업데이트하는 함수를 생성하였다.
<br></br>
> 10 epoch 만큼 모델 학습
```python
EPOCHS = 10
for epoch in range(1, EPOCHS+1):
    for i, images in enumerate(gan_datasets):
        gene_loss, disc_loss = gan_step(images)

        if (i+1) % 100 == 0:
            print(f"[{epoch}/{EPOCHS} EPOCHS, {i+1} ITER] G:{gene_loss}, D:{disc_loss}")
```
100 번의 반복마다 각 손실 (loss) 을 출력하도록 하였다.
<br></br>
> 학습된 모델 테스트
```python
import numpy as np

noise = tf.random.normal([10, 100])

output = gan_generator(noise)
output = np.squeeze(output.numpy())

plt.figure(figsize=(15,6))
for i in range(1, 11):
    plt.subplot(2,5,i)
    plt.imshow(output[i-1])
```
100차원 노이즈 입력을 10개 사용하여 10개의 숫자 손글씨 데이터를 생성해 시각화하였다.

경고메세지는 무시하여도 된다.

10 epoch 학습으로는 좋은 결과를 기대할 수 없었다.
<br></br>

원본과 같이 500 epoch 하기에는 학습 소요시간이 너무 오래 소요되므로 이미 학습된 가중치를 다운받아 사용해보자.

+ [GAN : 500 epoch 학습한 가중치](https://aiffelstaticprd.blob.core.windows.net/media/documents/GAN_500.zip)
<br></br>
>  GAN : 500 epoch 로 학습한 가중치 다운로드
```bash
$ mkdir -p ~/aiffel/conditional_generation/gan 
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/GAN_500.zip 
$ mv GAN_500.zip ~/aiffel/conditional_generation/gan 
$ cd ~/aiffel/conditional_generation/gan && unzip GAN_500.zip
```
<br></br>
> GAN : 500 epoch 로 학습한 가중치를 적용하여 테스트
```python
import os
weight_path = os.getenv('HOME')+'/aiffel/conditional_generation/gan/GAN_500'

noise = tf.random.normal([10, 100]) 

gan_generator = GeneratorGAN()
gan_generator.load_weights(weight_path)

output = gan_generator(noise)
output = np.squeeze(output.numpy())

plt.figure(figsize=(15,6))
for i in range(1, 11):
    plt.subplot(2,5,i)
    plt.imshow(output[i-1])
```
10 개의 서로 다른 숫자 이미지가 시각화 된다. (아닐 수도 있다.)

이런 방법으로는 원하는 특정 숫자를 출력하기 위해 몇 번의 입력을 넣어야할지 알 수 없다.
<br></br>

### cGAN 으로 MNIST 학습하기

> cGAN 으로 MNIST 학습
```python
cgan_generator = GeneratorCGAN()
cgan_discriminator = DiscriminatorCGAN()

@tf.function()
def cgan_step(real_images, labels):
    noise = tf.random.normal([real_images.shape[0], 100])
    
    with tf.GradientTape(persistent=True) as tape:
        fake_images = cgan_generator(noise, labels)
        
        real_out = cgan_discriminator(real_images, labels)
        fake_out = cgan_discriminator(fake_images, labels)
        
        gene_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)
    
    gene_grad = tape.gradient(gene_loss, cgan_generator.trainable_variables)
    disc_grad = tape.gradient(disc_loss, cgan_discriminator.trainable_variables)
    
    gene_opt.apply_gradients(zip(gene_grad, cgan_generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, cgan_discriminator.trainable_variables))
    return gene_loss, disc_loss


EPOCHS = 1
for epoch in range(1, EPOCHS+1):
    
    for i, (images, labels) in enumerate(cgan_datasets):
        gene_loss, disc_loss = cgan_step(images, labels)
    
        if (i+1) % 100 == 0:
            print(f"[{epoch}/{EPOCHS} EPOCHS, {i} ITER] G:{gene_loss}, D:{disc_loss}")
```
적은 epoch 를 통해서는 결과를 기대할 수 없기에 우선 1 epoch 만 학습하고, 위와 같이 원본처럼 500 epoch 로 학습된 가중치를 다운받아 적용하여 테스트 해보자.
<br></br>

> cGAN : 500 epoch 로 학습된 가중치 다운로드
```bash
$ mkdir -p ~/aiffel/conditional_generation/cgan 
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/CGAN_500.zip 
$ mv CGAN_500.zip ~/aiffel/conditional_generation/cgan 
$ cd ~/aiffel/conditional_generation/cgan && unzip CGAN_500.zip
```
+ 참고 : [ cGAN : 500 epoch 로 학습된 가중치](https://aiffelstaticprd.blob.core.windows.net/media/documents/CGAN_500.zip)
<br></br>
> cGAN : 500 epoch 로 학습된 가중치를 적용하여 테스트
```python
number =  7  # TODO : 생성할 숫자를 입력해 주세요!!

weight_path = os.getenv('HOME')+'/aiffel/conditional_generation/cgan/CGAN_500'

noise = tf.random.normal([10, 100])

label = tf.one_hot(number, 10)
label = tf.expand_dims(label, axis=0)
label = tf.repeat(label, 10, axis=0)

generator = GeneratorCGAN()
generator.load_weights(weight_path)

output = generator(noise, label)
output = np.squeeze(output.numpy())

plt.figure(figsize=(15,6))
for i in range(1, 11):
    plt.subplot(2,5,i)
    plt.imshow(output[i-1])
```
`number`라는 변수에 0 ~ 9 사이의 숫자 중 생성하길 원하는 숫자를 입력하여 실행한다.

경고 메세지가 출력된다면 무시해도 된다.

출력된 10 개의 시각화 이미지를 보면 `number` 에 입력한 숫자에 해당하는 손글씨가 시각화됨을 볼 수 있다.
<br></br>

## GAN 의 입력에 이미지를 넣는다면? Pix2Pix

Pix2Pix 는 존 노이즈 입력을 이미지로 변환하는 일반적인 GAN이 아니라 이미지를 입력으로 하여 원하는 다른 형태의 이미지로 변환시키는 GAN 모델의 일종이다.

구조는 cGAN 과 동일한 Conditional Adversarial Networks 을 사용한다.

+ 참고 : [논문 : Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/p2p_results.max-800x600.png)
<br></br>

위 그림은 Pix2Pix 논문에서 수행한 결과이다. 그림에서도 볼 수 있듯이 Pix2Pix 는 이미지 간의 변환을 의미한다.

그림에서 첫 번째 [Labels to Street Scene] 이미지는 픽셀별로 레이블 정보만 존재하는 segmentation map 을 입력으로 실제 거리 사진을 생성해 내었고, 이 외에 흑백 사진을 컬러로 변환하거나, 낮에 찍은 사진을 밤에 찍은 사진으로 변환하거나, 가방 스케치를 이용해 채색된 가방을 만들 수도 있음을 볼 수 있다.

한 이미지의 픽셀에서 다른 이미지의 픽셀로 (pixel to pixel) 변환한다는 뜻에서 Pix2Pix 라고 불리며, GAN 기반의 Image - to - Image Trsnlation 작업에서 가장 기초가되는 연구이다.

노이즈와 레이블 정보를 함께 입력했던 cGAN 은 fully-connected 레이어를 연속적으로 쌓아 만들었지만, 이미지 변환이 목적인 Pix2Pix 는 이미지를 다루는데 효율적인 convolution 레이어를 활용하며, GAN 구조를 기반으로하기 때문에 Generator와 Discriminator 두 가지 구성 요소로 이뤄진다.
<br></br>
### Pix2Pix (Generator)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/p2p_generator.max-800x600.png)
<br></br>
Generator는 어떠한 이미지를 입력받아 변환된 이미지를 출력하기 위해 사용되며, 입력 이미지와 변환된 이미지의 크기는 동일해야한다. 이런 문제에서 일반적으로 사용되는 구조는 위 그림과 같은 Encoder - Decoder 구조이다. 

Encoder에서 입력 이미지 $(x)$ 를 받으면 단계적으로 이미지를 down-sampling 하면서 입력 이미지의 중요한 representation을 학습한다.

Decoder 에서는 이를 이용해 반대로 다시 이미지를 up-sampling 하여 입력 이미지와 동일한 크기의 변환된 이미지 $(y)$ 를 생성한다.

이 과정은 모두 convolution 레이어로 진행되며, 레이어의 많은 파라미터를 학습하여 잘 변환된 이미지를 얻도록 한다.

여기서 Encoder의 최종 출력은 위 그림 중간에 위치한 가장 작은 사각형이며, `bottleneck` 이라고도 불리는 이 부분은 입력 이미지 $(x)$ 의 가장 중요한 특징만을 담고 있다.

이렇게 중요하지만 작은 특징이 변환된 이미지 $(y)$ 를 생성하는데 충분한 정보를 제공하기위해 논문에서는 U - Net 이라는 Generator 구조를 추가적으로 제안한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/p2p_generator_unet.max-800x600.png)
<br></br>
위 그림은 U - Net 구조를 나타낸 그림이다. 서로 마주보고 대칭을 이루는 형태를 가지고 있으며, 위에서 살펴본 단순한 Encoder - Decoder 로 구성된 Generator 와 달리 각 레이어마다 Encoder 와 Decoder 가 연결 (skip connection) 되어 있다.

이를 통해 Decoder 가 변환된 이미지를 더 잘 생성하도록 Encoder 로부터 더 많은 추가 정보를 이용할 수 있어 단순한 Encoder - Decoder 구조의 Generator 를 사용한 결과에 비해 비교적 선명한 결과를 얻을 수 있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/p2p_result_g.max-800x600.png)
<br></br>
위 그리미은 U - Net 을 사용한 Encoder-Decoder 구조의 Generator 의 결과물이다. 단순한 Encoder-Decoder 구조의 Generator 에 비해 훨씬 뛰어난 성능을 보여주고 있음을 확인할 수 있다.

+ 참고 : [U-Net 논문 리뷰](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a)
<br></br>

### Pix2Pix (Loss Function)

위 Generator 구조를 보면 Generator 만으로도 이미지 변환이 가능하지 않을까? 라는 생각을 할 수 있다.

출력된 이미지와 실제 이미지의 차이로 L2 (MSE), L1 (MAE) 같은 손실을 계산한 후 이를 역전파하여 네트워크를 학습시키면 당연히 변환하고자 하는 이미지를 Encoder에 입력하여 Decoder의 출력으로 변환된 이미지를 얻을 수 있다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/p2p_result_loss.max-800x600.png)
<br></br>

하지만 Generator 만을 이용한 이미지 변환의 문제점은 변환된 이미지의 품질이다.

위 그림에서 L1 이라 써있는 결과가 Generator 만을 사용해 변환된 이미지와 실제 이미지 사이의 L1 손실을 이용해 만들어낸 것이다.

이미지를 변환하는데 L1 (MAE) 이나 L2 (MSE) 손실만을 이용해서 학습하는 경우 이렇게 결과가 흐릿해지는 경향이 있으며, Generator 가 단순히 이미지의 평균적인 손실만을 줄이고자 파라미터를 학습하기 때문에 이러한 현상이 불가피하다.

하지만 cGAN 이라 쓰여진 GAN 기반의 학습 방법은 비교적 훨씬 더 세밀한 정보를 잘 표현하고 있음을 볼 수 있다. 이는 Discriminator 를 잘 속이려면 Generator 가 (Ground truth라고 쓰여진 이미지같이) 진짜 같은 이미지를 만들어야 하기 때문이다.

논문에서는 L1손실과 GAN 손실을 같이 사용하면 더욱더 좋은 결과를 얻을 수 있다고 한다. 
<br></br>

### Pix2Pix (Discriminator)

실제와 같은 이미지를 얻기 위해서는 GAN 학습 방법을 이용해야하며, 생성자의 라이벌인 판별자가 필요하다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/dcgan_d.png)
<br></br>
위 그림은 DCGAN의 Discriminator 의 구조를 나타낸 것이다.

DCGAN의 Discriminator는 생성된 가짜이미지 혹은 진짜이미지를 하나씩 입력받아 convolution 레이어를 이용해 점점 크기를 줄여나가면서, 최종적으로 하나의 이미지에 대해 하나의 확률값을 출력한다.

Pix2Pix 는 이 과정에서 하나의 전체 이미지에 대해 하나의 확률값만을 도출하는 것이 과연 진짜와 가짜를 판별하는 것이 좋을까? 라는 의문에서 시작되었다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/patchgan.max-800x600.png)
<br></br>
위 그림은 Pix2Pix 에서 사용되는 Discriminator 를 간략히 나타낸 것이다.

하나의 이미지가 Discriminator 의 입력으로 들어오면, convolution 레이어를 거쳐 확률값을 나타내는 최종 결과를 생성하며, 생성된 결과는 하나의 값이 아닌 여러 개의 값을 갖는다. (위 그림의 Prediction은 16개의 값을 가지고 있다.)

위 그림에서 입력이미지의 파란색 점선은 여러 개의 출력 중 하나의 출력을 계산하기 위한 입력이미지의 receptive field 영역을 나타내고 있으며, 전체 영역을 다 보는 것이 아닌 일부 영역 (파란색 점선) 에 대해서만 진짜 / 가짜를 판별하는 하나의 확률값을 도출한다.

이렇게 서로 다른 영역에 대해 진짜 / 가짜를 나타내는 여러 개의 확률값을 계산하고, 이 값을 평균하여 최종 Discriminator 의 출력을 생성한다. 이는 이미지의 일부 영역 (patch) 을 이용하기 때문에 PatchGAN 으로 불리며, 일반적으로 이미지에서 거리가 먼 두 픽셀은 서로 연관성이 거의 없기 때문에 특정 크기를 가진 일부 영역에 대해 세부적으로 진짜 / 가짜를 판별하는 것이 Generator 로 더 진짜 같은 이미지를 만들도록 하는 방법이다.
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/patchgan_results.max-800x600.png)
<br></br>

위 그림은 Pix2Pix 에서 사용되는 Discriminator 를 나타낸 그림에서 파란색 점선에 해당하는 판별 영역을 다양한 크기로 심헌한 결과를 나타낸것이다.

마지막에 보이는 286 x 286 (입력 이미지의 크기) 이라 적힌 이미지는 DCGAN 의 Discriminator 와 같이 전체 이미지에 대해 하나의 확률값을 출력하여 진짜 / 가짜를 판별하도록 학습한 결과이다.

70 x 70 이미지는 Discriminator 입력 이미지에서 70 x 70 크기를 갖는 일부 영역에 대해서 하나의 확률값을 출력한 것이며, 16 x 16, 1 x 1로 갈수록 더 작은 영역을 보고 각각의 확률값을 계산하므로 Discriminator 의 출력값의 개수가 더 많다.

위 그림을 살펴보면 너무 작은 patch 를 사용한 결과(1 x 1, 16 x 16)는 품질이 좋지 않으며, 70 x 70 patch 를 이용한 결과가 전체 이미지를 사용한 결과 (286 x 286) 보다 조금 더 사실적인 이미지를 생성하므로 PatchGAN 의 사용이 성공적이라고 볼 수 있다.
<br></br>

## 난 스케치를 할 테니 너는 채색을 하거라 (01). 데이터 준비하기

`Sketch2Pokemon`이라는 데이터셋을 활용해 Pix2Pix 모델을 구현해보자
<br></br>
> `Sketch2Pokemon` 다운
```bash
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/sketch2pokemon.zip 
$ mv sketch2pokemon.zip ~/aiffel/conditional_generation 
$ cd ~/aiffel/conditional_generation && unzip sketch2pokemon.zip
```
`Sketch2Pokemon` 데이터셋은 학습용 데이터 셋에 830 개의 이미지가 있으며, 각 (256 x 256) 크기의 이미지 쌍이 나란히 붙어 (256 x 512) 크기의 이미지로 구성되어 있다.

학습용 데이터셋만 따로 다운받았다.

+ 참고 :
	+ [Sketch2Pokemon info](https://www.kaggle.com/norod78/sketch2pokemon)
	+ [Sketch2Pokemon 데이터셋] (https://aiffelstaticprd.blob.core.windows.net/media/documents/sketch2pokemon.zip)
<br></br>
> 데이터 경로 설정
```python
import os

data_path = os.getenv('HOME')+'/aiffel/conditional_generation/pokemon_pix2pix_dataset/train/'
print("number of train examples :", len(os.listdir(data_path)))
```
830 개 이미지가 있음을 확인할 수 있다.
<br></br>
> 학습용 데이터셋에서 임의의 6 장의 이미지를 확인
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,15))
for i in range(1, 7):
    f = data_path + os.listdir(data_path)[np.random.randint(800)]
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    plt.subplot(3,2,i)
    plt.imshow(img)
```
하나의 이미지에 포켓몬 스케치와 실제 포켓몬 이미지가 함께 포함되어 있음을 확인할 수 있다.

일부는 스케치 되지 않은 이미지도 있음을 볼 수 있는데, 스케치 생셩 모델을 이용해 만든 데이터셋이기 때문이다.
<br></br>
> 이미지 하나의 크기를 확인
```python
f = data_path + os.listdir(data_path)[0]
img = cv2.imread(f, cv2.IMREAD_COLOR)
print(img.shape)
```
(256, 512, 3) 크기를 가지는 것을 확인할 수 있다.
<br></br>
> 모델 학습에 사용할 데이터를 (256, 256, 3) 크기의 2개 이미지로 분할
```python
import tensorflow as tf

def normalize(x):
    x = tf.cast(x, tf.float32)
    return (x/127.5) - 1

def denormalize(x):
    x = (x+1)*127.5
    x = x.numpy()
    return x.astype(np.uint8)

def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, 3)
    
    w = tf.shape(img)[1] // 2
    sketch = img[:, :w, :] 
    sketch = tf.cast(sketch, tf.float32)
    colored = img[:, w:, :] 
    colored = tf.cast(colored, tf.float32)
    return normalize(sketch), normalize(colored)

f = data_path + os.listdir(data_path)[1]
sketch, colored = load_img(f)

plt.figure(figsize=(10,7))
plt.subplot(1,2,1); plt.imshow(denormalize(sketch))
plt.subplot(1,2,2); plt.imshow(denormalize(colored))
```
<br></br>
> 학습의 다양성을 높이기 위해 augmentation 으로 데이터의 양을 증가시켜준다.
```python
from tensorflow import image
from tensorflow.keras.preprocessing.image import random_rotation

@tf.function() # 빠른 텐서플로 연산을 위해 @tf.function()을 사용합니다. 
def apply_augmentation(sketch, colored):
    stacked = tf.concat([sketch, colored], axis=-1)
    
    _pad = tf.constant([[30,30],[30,30],[0,0]])
    if tf.random.uniform(()) < .5:
        padded = tf.pad(stacked, _pad, "REFLECT")
    else:
        padded = tf.pad(stacked, _pad, "CONSTANT", constant_values=1.)

    out = image.random_crop(padded, size=[256, 256, 6])
    
    out = image.random_flip_left_right(out)
    out = image.random_flip_up_down(out)
    
    if tf.random.uniform(()) < .5:
        degree = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        out = image.rot90(out, k=degree)
    
    return out[...,:3], out[...,3:]   
```
`apply_augmentation` 함수는 스케치 및 채색된 2개 이미지를 입력으로 받아 여러 가지 연산을 두 이미지에 동일하게 적용한다.

정의된 `apply_augmentation` 함수는 두 이미지를 입력으로 받으며, 입력된 두 이미지는 (tf.concat) 에 의해 채널 축으로 연결된다. 만약 두 이미지가 각각 3 채널이라면 6 채널이 된다.

이렇게 채널 축으로 연결된 이미지에 (tf.pad) 에 의해 각 50 % 확률로 Refection padding 또는 constant padding 이 30 픽셀의 pad width 만큼적용된다.

이후 (tf.image.random_crop) 를 통해 (256, 256, 6) 크기를 가진 이미지를 임의로 잘라내고 (tf.image.random_flip_left_right) 를 통해 임의로 잘라낸 이미지를 50% 확률로 가로로 뒤집는다.

뒤집힌 이미지를 (tf.image.random_flip_up_down) 를 통해 다시 0% 확률로 세로로 뒤집고, 마지막으로 (tf.image.rot90) 를 통해 50% 확률로 회전시켜 Data Augmentation 한다.
<br></br>
> 위 Augmentation 함수를 적용시킨 데이터 이미지 시각화
```python
plt.figure(figsize=(15,13))
img_n = 1
for i in range(1, 13, 2):
    augmented_sketch, augmented_colored = apply_augmentation(sketch, colored)
    
    plt.subplot(3,4,i)
    plt.imshow(denormalize(augmented_sketch)); plt.title(f"Image {img_n}")
    plt.subplot(3,4,i+1); 
    plt.imshow(denormalize(augmented_colored)); plt.title(f"Image {img_n}")
    img_n += 1
```
매우 다양한 이미지가 생성됨을 확인할 수 있다.
<br></br>
> 위 과정들을 학습 데이터에 적용하고, 적용 여부를 시각화를 통해 확인
```python
from tensorflow import data

def get_train(img_path):
    sketch, colored = load_img(img_path)
    sketch, colored = apply_augmentation(sketch, colored)
    return sketch, colored

train_images = data.Dataset.list_files(data_path + "*.jpg")
train_images = train_images.map(get_train).shuffle(100).batch(4)

sample = train_images.take(1)
sample = list(sample.as_numpy_iterator())
sketch, colored = (sample[0][0]+1)*127.5, (sample[0][1]+1)*127.5

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(sketch[0].astype(np.uint8))
plt.subplot(1,2,2); plt.imshow(colored[0].astype(np.uint8))
```
<br></br>

## 난 스케치를 할 테니 너는 채색을 하거라 (02). Generator 구성하기

Tensorflow의 Subclassing 방법을 통해 Pix2Pix 를 구현해 보자.
<br></br>
### Generator의 구성요소 알아보기

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/paper_g.png)
<br></br>

위 그림은 논문에서 Pix2Pix 의 Generator 를 구성하는데 필요한 정보가 담긴 부분을 발췌한 것이다.

논문에서 표기된 encoder 의 C64 는 순서대로 64 개의 4 x 4 필터에 stride 2 를 적용한 Convolution 와 0.2 slope 의 LeakyReLU 레이어의 조합을 나타낸다.

또한 decoder 의 CD512 는 순서대로 512개의 4x4 필터에 stride 2를 적용한 (Transposed) Convolution 와 BatchNorm , 50% Dropout, ReLU 레이어들의 조합을 나타낸다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/refer_g.max-800x600.png)
<br></br>
위 그림에서 `ENCODE` , `DECODE` 라고 쓰인 각각의 블록을 기준으로 양쪽에 쓰인 입출력 크기를 보면 "in" 이라고 쓰여진 입력 부분부터 윗줄의 화살표를 쭉 따라가면 계산된 결과의 (width, height) 크기가 점점 절반씩 줄어들며 최종적으로 (1, 1)이 되고, 채널의 수는 512 까지 늘어나는 것을 확인할 수 있다.

처음 입력부터 시작해서 (1, 1, 512) 크기를 출력하는 곳까지가 Encoder 부분이이다.

또한 아랫줄 화살표를 따라가면 (width, height) 크기가 점점 두 배로 늘어나 다시 (256, 256) 크기가 되고, 채널의 수는 점점 줄어들어 처음 입력과 같이 3 채널이 되는데 (1, 1, 512) 를 입력으로 최종 출력까지의 연산들이 Decoder 부분이다.
<br></br>

### Generator 구현하기

> Generator 의 Encoder 에 사용할 블록 구현
```python
from tensorflow.keras import layers, Input, Model

class EncodeBlock(layers.Layer):
    def __init__(self, n_filters, use_bn=True):
        super(EncodeBlock, self).__init__()
        self.use_bn = use_bn       
        self.conv = layers.Conv2D(n_filters, 4, 2, "same", use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.lrelu= layers.LeakyReLU(0.2)

    def call(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.batchnorm(x)
        return self.lrelu(x)
```
논문에서 "C64", "C128" 등으로 쓰여진 것과 같이 "Convolution → BatchNorm → LeakyReLU" 의 3개 레이어로 구성된 기본적인 블록을 아래와 같이 하나의 레이어로 만들었다.

`__init__()` 메서드에서 `n_filters`, `use_bn` 를 설정하여 사용할 필터의 개수와 BatchNorm 사용 여부를 결정 할 수 있다.

이외 Convolution 레이어에서 필터의 크기(=4) 및 stride(=2) 와 LeakyReLU 활성화의 slope coefficient( = 0.2) 는 모든 곳에서 고정되어 사용하므로 각각의 값을 지정해 주었다.
<br></br>
> Encoder 블록을 사용해 Generator 의 Encoder 구성
```python
class Encoder(layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        filters = [64,128,256,512,512,512,512,512]
        
        self.blocks = []
        for i, f in enumerate(filters):
            if i == 0:
                self.blocks.append(EncodeBlock(f, use_bn=False))
            else:
                self.blocks.append(EncodeBlock(f))
    
    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
    def get_summary(self, input_shape=(256,256,3)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()
```
각 블록을 거치면서 사용할 필터의 개수를 `filters` 라는 리스트에 지정해 두었으며, `blocks` 이라는 리스트에는 사용할 블록들을 정의해 넣어두고, `call()` 메서드에서 차례대로 블록들을 통과합니다. 앞서 퀴즈로 알아본 것처럼 Encoder 첫 번째 블록에서는 BatchNorm 을 사용하지 않는다.

`get_summary` 는 레이어가 제대로 구성되었는지 확인하기 위한 용도로 따로 만들어 놓았다.
<br></br>
> Encoder 입력시 출력 확인
```python
Encoder().get_summary()
```
Encoder 에 (256, 256, 3) 크기의 데이터를 입력했을 때, 어떤 크기의 데이터가 출력되는지 살펴보자.

블록을 통과할수록 (width, height) 크기는 반씩 줄어들고, 사용된 필터의 수는 최대 512 개로 늘어나 최종 (1, 1, 512)로 알맞은 크기가 출력됨을 확인 할 수 있다.
<br></br>
> Generator 의 Decoder 에서 사용할 기본 블록 구현
```python
class DecodeBlock(layers.Layer):
    def __init__(self, f, dropout=True):
        super(DecodeBlock, self).__init__()
        self.dropout = dropout
        self.Transconv = layers.Conv2DTranspose(f, 4, 2, "same", use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
    def call(self, x):
        x = self.Transconv(x)
        x = self.batchnorm(x)
        if self.dropout:
            x = layers.Dropout(.5)(x)
        return self.relu(x)

    
class Decoder(layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        filters = [512,512,512,512,256,128,64]
        
        self.blocks = []
        for i, f in enumerate(filters):
            if i < 3:
                self.blocks.append(DecodeBlock(f))
            else:
                self.blocks.append(DecodeBlock(f, dropout=False))
                
        self.blocks.append(layers.Conv2DTranspose(3, 4, 2, "same", use_bias=False))
        
    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x
            
    def get_summary(self, input_shape=(1,1,256)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()
```
처음 세 개의 블록에서만 `Dropout` 을 사용했으며, 마지막 convolution 에는 3 개의 필터를 사용해 출력하는 것을 확인할 수 있다.
<br></br>
> Decoder 블록에 입력값을 넣을 을 때 출력값 확인
```python
Decoder().get_summary()
```
(width, height) 크기가 점점 늘어나고 사용 필터의 수는 점점 줄어들어 최종 (256, 256, 3) 크기로 알맞게 출력됨을 알 수 있다.
<br></br>

이렇게 구성한 Encoder 와 Decoder 를 연결시키면 Encoder 에서 (256, 256, 3) 입력이 (1, 1, 512) 로 변환되고, Decoder 를 통과해 다시 원래 입력 크기와 같은 (256, 256, 3)의 결과를 얻을 수 있으며, 스케치를 입력으로 이런 연산 과정을 통해 채색된 이미지 출력을 얻을 수 있다.

<br></br>

> Encoder와 Decoder를 연결해 Generator를 구성
```python
class EncoderDecoderGenerator(Model):
    def __init__(self):
        super(EncoderDecoderGenerator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
   
    def get_summary(self, input_shape=(256,256,3)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()
        

EncoderDecoderGenerator().get_summary()
```
`tf.keras.Model` 을 상속받아 Encoder 와 Decoder 를 연결해 Generator 를 구성할 수 있으며, Generator 를 잘 작동시키기 위해서는 약 4000 만 개의 파라미터를 잘 학습시켜야 함을 볼 수 있다.

<br></br>

## 난 스케치를 할 테니 너는 채색을 하거라 (03). Generator 재구성하기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/p2p_result_g2.max-800x600.png)
<br></br>
Pix2Pix 의 Generator 구조는 위 그림과 같이 2 가지를 제안했다.

위 그림의 각 구조 아래 표시된 이미지는 해당 구조를 Generator 로 사용했을 때의 결과이다. 단순한 Encoder-Decoder 구조에 비해 Encoder 와 Decoder 사이를 skip connection 으로 연결한 U - Net 구조를 사용한 결과가 훨씬 더 실제 이미지에 가까운 품질을 보인다.

앞서 구현한 Generator 는 위 그림의 Encoder-decoder  구조와 동일하며, Encoder 에서 출력된 결과를 Decoder 의 입력으로 연결했고, 이 외에 추가적으로 Encoder 와 Decoder 를 연결시키는 부분은 없다.

더 좋은 결과를 위해 앞서 구현한 것들을 조금씩 수정하여 U - Net 구조를 만들어보자.
<br></br>
> 앞서 구현한 Encoder 및 Decoder 에 사용되는 기본블록 가져오기
```python
class EncodeBlock(layers.Layer):
    def __init__(self, n_filters, use_bn=True):
        super(EncodeBlock, self).__init__()
        self.use_bn = use_bn       
        self.conv = layers.Conv2D(n_filters, 4, 2, "same", use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.lrelu = layers.LeakyReLU(0.2)

    def call(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.batchnorm(x)
        return self.lrelu(x)

    
class DecodeBlock(layers.Layer):
    def __init__(self, f, dropout=True):
        super(DecodeBlock, self).__init__()
        self.dropout = dropout
        self.Transconv = layers.Conv2DTranspose(f, 4, 2, "same", use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
    def call(self, x):
        x = self.Transconv(x)
        x = self.batchnorm(x)
        if self.dropout:
            x = layers.Dropout(.5)(x)
        return self.relu(x)
```
Encoder 및 Decoder 부분에서는 수정할 사항이 없으므로 그대로 사용한다.
<br></br>
> 사전에 정의된 블록을 이용해 U - Net Generator 정의
```python
class UNetGenerator(Model):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        encode_filters = [64,128,256,512,512,512,512,512]
        decode_filters = [512,512,512,512,256,128,64]
        
        self.encode_blocks = []
        for i, f in enumerate(encode_filters):
            if i == 0:
                self.encode_blocks.append(EncodeBlock(f, use_bn=False))
            else:
                self.encode_blocks.append(EncodeBlock(f))
        
        self.decode_blocks = []
        for i, f in enumerate(decode_filters):
            if i < 3:
                self.decode_blocks.append(DecodeBlock(f))
            else:
                self.decode_blocks.append(DecodeBlock(f, dropout=False))
        
        self.last_conv = layers.Conv2DTranspose(3, 4, 2, "same", use_bias=False)
    
    def call(self, x):
        features = []
        for block in self.encode_blocks:
            x = block(x)
            features.append(x)
        
        features = features[:-1]
                    
        for block, feat in zip(self.decode_blocks, features[::-1]):
            x = block(x)
            x = layers.Concatenate()([x, feat])
        
        x = self.last_conv(x)
        return x
                
    def get_summary(self, input_shape=(256,256,3)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()
```
모델의 `__init__()` 메서드에서 Encoder 및 Decoder 에서 사용할 모든 블록들을 정의해 놓고, `call()` 에서 forward propagation 하도록 한다.

이전 구현에는 없었던 skip connection 이 `call()` 내부에서 어떻게 구현되었는지 잘 확인해보자.

먼저, `__init__()` 에서 정의된 `encode_blocks` 및 `decode_blocks` 가 `call()` 내부에서 차례대로 사용되어 Encoder 및 Decoder 내부 연산을 수행하며, `call()` 내부의 `features = features[:-1]` 가 필요한 이유는 Skip connection 을 위해 만들어진 features 리스트에는 Encoder 내 각 블록의 출력이 들어있는데, Encoder 의 마지막 출력 (feature 리스트 의 마지막 항목) 은 Decoder 로 직접 입력되므로 skip connection 의 대상이 아니기 때문이다.

또한 `features[::-1]` 을 사용한 이유는 Skip connection 은 Encoder 내 첫 번째 블록의 출력이 Decoder 의 마지막 블록에 연결되고, Encoder 내 두 번째 블록의 출력이 Decoder 의 뒤에서 2 번째 블록에 연결되는 등 대칭을 이룬다. 

features 에는 Encoder 블록들의 출력들이 순서대로 쌓여있고, 이를 Decoder 에서 차례대로 사용하기 위해서 features 의 역순으로 연결해야하기 때문에 필요하다.

이때, Encoder 와 Decoder 사이의 skip connection 을 위해 `features` 라는 리스트를 만들고 Encoder 내에서 사용된 각 블록들의 출력을 차례로 담는다.

이후 Encoder 의 최종 출력이 Decoder 의 입력으로 들어가면서 다시 한번 각각의 Decoder 블록들을 통과하는데,  
`features` 리스트에 있는 각각의 출력들이 Decoder  블록 연산 후 함께 연결되어 다음 블록의 입력으로 사용된다.

예를 들어  `데이터 A 크기 : (32,128,128,200) #(batch, width, height, channel)` 와 `데이터 B 크기 : (32,128,128,400) #(batch, width, height, channel)` 가 있으 ㄹ때 skip connection 의 layers.Concatenate() 결과는 layers.Concatenate() 내에 별다른 설정이 없다면 가장 마지막 축(채널 축)을 기준으로 서로 연결되므로 (128, 128, 600) 이 된다.
<br></br>
> U-Net 구조 Generator 내부의 각 출력이 적절한지 확인
```python
UNetGenerator().get_summary()
```
이전 Encoder - Decoder Generator 구조에서 학습해야 할 파라미터는 약 4000 만 개였던 반면 Skip connection 을 추가한 U - Net Generator 의 파라미터는 약 5500 만 개로 증가하였다.

이때 각 convolution 레이어에서 사용된 필터의 수는 두 종류의 Decoder에서 동일하지만, 그 크기가 다르다.

예를 들어, 이전 Decoder 블록의 출력의 크기가 (16, 16, 512)라면,  
Encoder - decoder Generator 의 경우, Decoder 의 다음 블록에서 계산할 convolution 의 필터 크기는 4 x 4 x 512 이다. U - Net Generator 의 경우, Encoder 내 블록 출력이 함께 연결되어 Decoder 의 다음 블록에서 계산할 convolution 의 필터 크기는 4 x 4 x (512 + 512) 가 된다.
<br></br>

이를 정리하자면, U-Net Generator 에서 사용한 skip-connection 으로 인해 Decoder 의 각 블록에서 입력받는 채널 수가 늘어났고, 이에 따라 블록 내 convolution 레이어에서 사용하는 필터 크기가 커지면서 학습해야 할 파라미터가 늘어났다.
<br></br>

## 난 스케치를 할 테니 너는 채색을 하거라 (04). Discriminator

Generator 만을 이용한 결과보다 더 사실적인 이미지를 생성하기 위해 Discriminator 이 필요하다. 이 Discriminator 을 구현해 보자.
<br></br>

### Discriminator 의 구성요소 알아보기

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/paper_d.png)
<br></br>
위 그림은 Pix2Pix 의 Discriminator 를 구성하는데 필요한 정보를 발췌해 온 것이다.

Generator의 구성 요소와 동일하게 "C64" 등으로 표기되어있으며, 진짜 및 가짜 이미지를 판별하기 위해 최종 출력에 sigmoid 를 사용하는 것을 제외하면 특별한 변경 사항은 없다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/refer_d.max-800x600.png)
<br></br>

위 그림은 Discriminator 의 구조를 보다 자세히 나타낸 것이다.

Discriminator 는 2 개 입력(위 그림의 "in", "unknown")을 받아 연결 (CONCAT) 한 후, `ENCODE` 라고 쓰인 5 개의 블록을 통과한다. 이 중 마지막 블록을 제외한 4개 블록은 위 논문에서 표기된 "C64 - C128 - C256 - C512" 에 해당하며, 마지막은 1 (채널) 차원 출력을 위한 블록이 추가되었다.

최종적으로 출력되는 크기는 (30, 30, 1) 이며, 위 그림의 출력 이전의 2 개의 `ENCODE` 블록을 보면 각각의 출력 크기가 32, 31, 30 으로 1 씩 감소하는 것을 알 수 있다.

Generator 에서도 사용했던 2 stride convolution 에 패딩을 이용하면 (width, height) 크기가 절반씩 감소할 것이며, 1 stride convolution에 패딩을 하지 않는다면 (width, height) 크기는 필터 크기가 4 이므로 3 씩 감소한다.

그림과 같이 1 씩 감소하도록 하기위해서는 아래의 코드를 보며 알아보도록 한다.

추가적으로 위 그림에서 최종 출력 크기가 (30, 30)이 되어야 하는 이유는 앞서 Discriminator 에 대해 알아봤던 70 x 70 PatchGAN 을 사용했기 때문이다. 

최종 (30, 30) 출력에서 각 픽셀의 receptive field 크기를 (70, 70)으로 맞추기 위해 Discriminator 의 출력 크기를 (30, 30) 크기로 강제로 맞추는 과정이다.

+ 참고 : [Understanding PatchGAN](https://medium.com/@sahiltinky94/understanding-patchgan-9f3c8380c207)
<br></br>

### Discriminator 구현하기

> Discriminator 에 사용할 기본 블록 구형
```python
class DiscBlock(layers.Layer):
    def __init__(self, n_filters, stride=2, custom_pad=False, use_bn=True, act=True):
        super(DiscBlock, self).__init__()
        self.custom_pad = custom_pad
        self.use_bn = use_bn
        self.act = act
        
        if custom_pad:
            self.padding = layers.ZeroPadding2D()
            self.conv = layers.Conv2D(n_filters, 4, stride, "valid", use_bias=False)
        else:
            self.conv = layers.Conv2D(n_filters, 4, stride, "same", use_bias=False)
        
        self.batchnorm = layers.BatchNormalization() if use_bn else None
        self.lrelu = layers.LeakyReLU(0.2) if act else None
        
    def call(self, x):
        if self.custom_pad:
            x = self.padding(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
                
        if self.use_bn:
            x = self.batchnorm(x)
            
        if self.act:
            x = self.lrelu(x)
        return x 
```
`__init__()` 에서 필요한 만큼 많은 설정을 가능하게끔 했으며, 필터의 수(`n_filters`), 필터가 순회하는 간격(`stride`), 출력 feature map 의 크기를 조절할 수 있도록 하는 패딩 설정(`custom_pad`), BatchNorm 의 사용 여부(`use_bn`), 활성화 함수 사용 여부(`act`)가 설정이 가능하다.

만약 `DisBlock` 설정을 `DiscBlock(n_filters=64, stride=1, custom_pad=True, use_bn=True, act=True)` 으로 하여 생성된 블록에 (width, height, channel) = (128, 128, 32) 크기가 입력된다면 블록 내부에서는 가장 먼저 패딩 레이어인 `layers.ZeroPadding2D()` 를 통과하며 (130, 130, 32) 크기로 변한다.

다음으로 Convolution 레이어인 `layers.Conv2D(64,4,1,"valid")` 를 통과하여 (127, 127, 64) 크기로 변하며, 다음으로 BatchNormalization 레이어인 `layers.BatchNormalization()` 를 통과하여 (127, 127, 64) 크기로 변한다.

마지막으로 LeakyReLU 활성화 레이러인 `layers.LeakyReLU(0.2)` 를 통과하며 (127, 127, 64) 크기로 출력된다.

위에서 Discriminator 를 표현한 그림에서 마지막 2 개블록의 출력은 입력에 비해 (width, height) 크기가 1 씩 감소한다고 하였다.

(width, height) 크기가 1 씩 감소하게 하기위한 방법은 먼저 (128,128,32) 크기의 입력이 `layers.ZeroPadding2D()` 를 통과하면, width 및 height 의 양쪽 면에 각각 1 씩 패딩되어 총 2 만큼 크기가 늘어나 (130, 130, 32) 의 출력을 가진다.

이때 패딩하지 않고 필터 크기 4 및 간격 (stride) 1 의 convolution 레이어를 통과하면 width 및 height 가 3 씩 줄어들며 채널 수는 사용한 필터의 개수와 같아져 (127, 127, 64) 가 출력으로 된다.

(이는 `OutSize = (InSize + 2 ∗ PadSize − FilterSize) / Stride + 1` 의 식으로 계산할 수 있다.)

이 외에 다른 레이어(BatchNorm, LeakyReLU)는 출력의 크기에 영향을 주지 않기 때문에 1 씩 감소한다.
<br></br>
> 마지막 2 개블록의 출력은 입력에 비해 (width, height) 크기가 1 씩 감소하는 과정을 각 출력의 크기가 맞는지 코드를 통해 확인
```python
inputs = Input((128,128,32))
out = layers.ZeroPadding2D()(inputs)
out = layers.Conv2D(64, 4, 1, "valid", use_bias=False)(out)
out = layers.BatchNormalization()(out)
out = layers.LeakyReLU(0.2)(out)

Model(inputs, out).summary()
```
코드와 비슷한 설정으로 (width, height) 크기를 1 씩 감소시킬 수 있으며, 마지막 2개 블록은 출력의 크기가 1씩 감소하므로 이런 방식을 적용하면 된다.
<br></br>
> Discriminator 구성
```python
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.block1 = layers.Concatenate()
        self.block2 = DiscBlock(n_filters=64, stride=2, custom_pad=False, use_bn=False, act=True)
        self.block3 = DiscBlock(n_filters=128, stride=2, custom_pad=False, use_bn=True, act=True)
        self.block4 = DiscBlock(n_filters=256, stride=2, custom_pad=False, use_bn=True, act=True)
        self.block5 = DiscBlock(n_filters=512, stride=1, custom_pad=True, use_bn=True, act=True)
        self.block6 = DiscBlock(n_filters=1, stride=1, custom_pad=True, use_bn=False, act=False)
        self.sigmoid = layers.Activation("sigmoid")
        
        # filters = [64,128,256,512,1]
        # self.blocks = [layers.Concatenate()]
        # for i, f in enumerate(filters):
        #     self.blocks.append(DiscBlock(
        #         n_filters=f,
        #         strides=2 if i<3 else 1,
        #         custom_pad=False if i<3 else True,
        #         use_bn=False if i==0 and i==4 else True,
        #         act=True if i<4 else False
        #     ))
    
    def call(self, x, y):
        out = self.block1([x, y])
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        return self.sigmoid(out)
    
    def get_summary(self, x_shape=(256,256,3), y_shape=(256,256,3)):
        x, y = Input(x_shape), Input(y_shape) 
        return Model((x, y), self.call(x, y)).summary()
```
`__init__()` 내부에서 사용할 블록들을 정의했는데, 이전의 구현들처럼 (위 코드의 주석 처리된 부분과 같이) for loop 로 간편하게 블록을 만들 수도 있지만, 쉽게 코드를 읽게끔 총 6 개 블록을 각각 따로 만들었다.

첫 번째 블록은 단순한 연결(`concat`) 을 수행하며, Discriminator 의 최종 출력은 `sigmoid` 활성화 함수를 사용했다.
<br></br>
> 각 블록의 출력 크기 확인
```python
Discriminator().get_summary()
```
두 개의 (256, 256, 3) 크기 입력으로 최종 (30, 30, 1) 출력을 만들었다.
<br></br>
> 임의의 (256, 256, 3) 크기의 입력을 넣었을 때 나오는(30, 30) 출력 시각화
```python
x = tf.random.normal([1,256,256,3])
y = tf.random.uniform([1,256,256,3])

disc_out = Discriminator()(x, y)
plt.imshow(disc_out[0, ... ,0])
plt.colorbar()
```
위 (30, 30) 크기를 갖는 결과 이미지의 각 픽셀값은 원래 입력의 (70, 70) 패치에 대한 분류 결과이다.

전체 입력의 크기가 (256, 256) 이므로, 각각의 (70, 70) 패치는 원래 입력상에서 많이 겹쳐있으며, 각각의 픽셀값은 `sigmoid` 함수의 결괏값이므로 0 ~ 1 사이의 값을 가지며, 진짜 및 가짜 데이터를 판별해내는 데 사용한다.
<br></br>
## 난 스케치를 할 테니 너는 채색을 하거라 (05). 학습 및 테스트하기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/p2p_result_loss2.max-800x600.png)
<br></br>
위 그림은 앞서 구현한 Generator 와 Discriminator 를 학습에 사용되는 손실함수에 따른 차이를 나타낸 것이다.

레이블 정보만 있는 입력에 대해 여러 손실함수를 사용해 실제 이미지를 만들어 낸 결과는, 일반적인 GAN 의 손실함수에 L1 을 추가로 이용했을 때 가장 실제에 가까운 이미지를 생성한다.
<br></br>
> 손실함수 설정
```python
from tensorflow.keras import losses

bce = losses.BinaryCrossentropy(from_logits=False)
mae = losses.MeanAbsoluteError()

def get_gene_loss(fake_output, real_output, fake_disc):
    l1_loss = mae(real_output, fake_output)
    gene_loss = bce(tf.ones_like(fake_disc), fake_disc)
    return gene_loss, l1_loss

def get_disc_loss(fake_disc, real_disc):
    return bce(tf.zeros_like(fake_disc), fake_disc) + bce(tf.ones_like(real_disc), real_disc)
```
두 가지 손실 함수를 모두 사용하며 Generator 와 Discriminator 의 손실 계산은 다음과 같다.

Generator 의 손실함수 (위 코드의 `get_gene_loss`)는 총 3개의 입력이 있으며, 이 중 `fake_disc`는 Generator가 생성한 가짜 이미지를 Discriminator 에 입력하여 얻어진 값이며, 실제 이미지를 뜻하는 "1" 과 비교하기 위해 `tf.ones_like()` 를 사용한다.

또한 L1 손실을 계산하기 위해 생성한 가짜 이미지(`fake_output`)와 실제 이미지(`real_output`) 사이의 MAE (Mean Absolute Error) 를 계산한다.

Discriminator 의 손실함수 (위 코드의 `get_disc_loss`)는 2 개의 입력이 있으며, 이들은 가짜 및 진짜 이미지가 Discriminator 에 각각 입력되어 얻어진 값이다. 

Discriminator 는 실제 이미지를 잘 구분해 내야 하므로 `real_disc`는 "1" 로 채워진 벡터와 비교하고, `fake_disc`는 "0" 으로 채워진 벡터와 비교한다.
<br></br>
> optimizer 설정
```python
from tensorflow.keras import optimizers

gene_opt = optimizers.Adam(2e-4, beta_1=.5, beta_2=.999)
disc_opt = optimizers.Adam(2e-4, beta_1=.5, beta_2=.999)
```
논문과 동일하게 사용한다.
<br></br>
> 하나의 배치 크기만큼 데이터를 입력했을 때 가중치를 1회 업데이트하는 과정 구현
```python
@tf.function
def train_step(sketch, real_colored):
    with tf.GradientTape() as gene_tape, tf.GradientTape() as disc_tape:
        # Generator 예측
        fake_colored = generator(sketch, training=True)
        # Discriminator 예측
        fake_disc = discriminator(sketch, fake_colored, training=True)
        real_disc = discriminator(sketch, real_colored, training=True)
        # Generator 손실 계산
        gene_loss, l1_loss = get_gene_loss(fake_colored, real_colored, fake_disc)
        gene_total_loss = gene_loss + (100 * l1_loss) ## <===== L1 손실 반영 λ=100
        # Discrminator 손실 계산
        disc_loss = get_disc_loss(fake_disc, real_disc)
                
    gene_gradient = gene_tape.gradient(gene_total_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    gene_opt.apply_gradients(zip(gene_gradient, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
    return gene_loss, l1_loss, disc_loss
```
cGAN 과 전반적으로 동일한 학습 과정을 가지지만, 위 코드의 `gene_total_loss` 계산 라인에서 최종 Generator 손실을 계산할 때, L1 손실에 100 을 곱한 부분 (코드에서 ## 표시가 있는 부분) 은 Generator 의 손실에 해당하며 논문에서는 다음 수식을 통해 정의하였다.

$$G^* = \arg \min_G \max_D \mathcal{L}{cGAN}(G,D) + \lambda \mathcal{L}{L1}(G)$$ 

위 식에서 $λ$는 학습 과정에서 L1 손실을 얼마나 반영할 것인지를 나타내며 논문에서는 $λ = 100$ 을 사용하였다.
<br></br>
> 모델 학습
```python
EPOCHS = 10

generator = UNetGenerator()
discriminator = Discriminator()

for epoch in range(1, EPOCHS+1):
    for i, (sketch, colored) in enumerate(train_images):
        g_loss, l1_loss, d_loss = train_step(sketch, colored)
                
        # 10회 반복마다 손실을 출력합니다.
        if (i+1) % 10 == 0:
            print(f"EPOCH[{epoch}] - STEP[{i+1}] \
                    \nGenerator_loss:{g_loss.numpy():.4f} \
                    \nL1_loss:{l1_loss.numpy():.4f} \
                    \nDiscriminator_loss:{d_loss.numpy():.4f}", end="\n\n")
```
학습에 오랜 시간이 소요되므로 10 epoch 만 학습시켜보았다.
<br></br>
> 모델 테스트
```python
test_ind = 1

f = data_path + os.listdir(data_path)[test_ind]
sketch, colored = load_img(f)

pred = generator(tf.expand_dims(sketch, 0))
pred = denormalize(pred)

plt.figure(figsize=(20,10))
plt.subplot(1,3,1); plt.imshow(denormalize(sketch))
plt.subplot(1,3,2); plt.imshow(pred[0])
plt.subplot(1,3,3); plt.imshow(denormalize(colored))
```
썩 마음에 드는 결과는 아니지만 Pix2Pix 로 128 epoch 학습 후 테스트 결과가 아래와 같다고한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/pokemon_result.max-800x600.png)
<br></br>
10 epoch 학습의 결과보다는 뛰어나지만 전체적인 색감 정도의 향상 외에 세부적으로는 제대로 채색되지 않는다는 것을 확인할 수 있다.
<br></br>
