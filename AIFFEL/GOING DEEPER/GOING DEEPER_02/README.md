# 02. 없다면 어떻게 될까? (ResNet Ablation Study)

딥러닝 논문은 여러 방법들을 적용하며, 적용 전과 후의 차이를 실험을 통해 결과로 보여준다.

딥러닝 논문의 모델을 구현해보고, 모델에 각 기법을 적용했을 때와 아닐 때를 비교해보며 효과를 체감해보자.


## 실습목표

1.  직접 ResNet 구현하기
2.  모델을 config에 따라서 변경가능하도록 만들기
3.  직접 실험해서 성능 비교하기


## 학습내용

1.  Ablation Study
2.  Back to the 2015
3.  Block
4.  Complete Model
5.  Experiment

<br></br>
## Ablation Study

애블레이션 연구 (Ablation Study) 는 제거하는 연구로 직역할 수 있다.

많은 딥러닝 논문에서 기존의 방법에 여러 가지 시도를 통해 당면한 문제를 해결하고자 한다.

이렇게 시도한 방법들이 성능 개선에 유효함을 증명하는 방법으로 새롭게 시도한 모델의 실험 결과와, 새롭게 시도한 부분을 제거한 모델로 실험한 결과를 비교한다.

즉, 어떠한 효과 / 방법을 주었을 때와 주지 않았을 때. 전 / 후를 비교함으로써 그 효과를 입증한다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-1-P-2_KU8V5aZ.max-800x600.png)

위 이미지는 ResNet 논문에서 제시한 residual connection 의 애블레이션 연구이다.

ImageNet 검증 데이터셋에 대한 Top - 1 error rate 를 지표로 사용하였으며, residual connection 유무를 통해 ResNet 을 비교하였다.

이때 residual connection 이 적용되지 않은 모델을 plain net 이라고 하였다.

이러한 애블레이션 연구를 통해 residual connection 을 활용한 Deep Network 가 보다 나은 성능을 보여주고 있음을 증명하였다.

<br></br>
## Ablation Study 실습 (1). CIFAR - 10 데이터셋 준비하기

ResNet 의 성능을 평가하기위해 다른 데이터셋에 적용하여 얼마만큼의 성능을 보이는지 확인해보자.

CIFAR - 10 데이터셋을 활용하여 일반 네트워크와 ResNet 를 비교하여 유효성을 확인해보자.

* 참고 : [ResNet 이론 - Deep residual learning for image recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

<br></br>
### 1. CIFAR -10

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/gc-1-p-cifar-10.png)

CIFAR - 10 은 10개의 카테고리에 해당하는 6만장의 이미지가 있으며, 각 이미지의 크기는 32 픽셀이다. 즉, 32 x 32 x 3 크기의 이미지를 가진다.

* 참고 : [CIFAR - 10](https://www.tensorflow.org/datasets/catalog/cifar10)

<br></br>
> 데이터셋 준비 및 필요 라이브러리 가져오기
> (`tensorflow-datasets` 를 통해 데이터셋 다운)
```python
pip install tensorflow-datasets

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

# Tensorflow가 활용할 GPU가 장착되어 있는지 확인해 봅니다.
tf.config.list_physical_devices('GPU')
```
`tfds.load()` 는 기본적으로 `~/tensorflow_datasets` 경로에 데이터셋을 다운한다. 저장 경로를 바꿀때는 `data_dir` 인자를 사용한다.
<br></br>
> 데이터 가져오기
```python
import urllib3
urllib3.disable_warnings()

#tfds.disable_progress_bar()   # 이 주석을 풀면 데이터셋 다운로드과정의 프로그레스바가 나타나지 않습니다.

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)
```
<br></br>
> 불러온 데이터셋의 특징 정보 확인
```python
# Tensorflow 데이터셋을 로드하면 꼭 feature 정보를 확인해 보세요. 
print(ds_info.features)
```
<br></br>
> 데이터 개수 확인
```python
# 데이터의 개수도 확인해 봅시다. 
print(tf.data.experimental.cardinality(ds_train))
print(tf.data.experimental.cardinality(ds_test))
```
Tensorflow 데이터셋은 넘파이가 아니기때문에 특징 정보와 개수 등을 꼭 확인하는 버릇을 들이도록 하자.
<br></br>


### Input Normalization

이미지는 픽셀을 통해 색상을 표현하며 RGB, Gray Scale, HSV, CMYK 등 다양한 색상 표현 체계를 가진다.

이미지를 모델이 넣기전에 모델이 어떤 이미지를 받는지를 고려해야하며 일반적으로 정규화 (Normalize) 를 통해 이미지를 0 ~ 1 사이의 값을 가지도록 바꿔준뒤 모델 학습을 수행한다.

- 참고 : [딥러닝 용어 정리, Normalization(정규화) 설명](https://light-tree.tistory.com/132)

<br></br>
> 이미지 정규화를 위한 함수 생성
```python
def normalize_and_resize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    # image = tf.image.resize(image, [32, 32])
    return tf.cast(image, tf.float32) / 255., label
```
<br></br>
> 데이터 셋에 정규화를 수행하는 함수 생성
```python
def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(
        normalize_and_resize_img, 
        num_parallel_calls=1
    )
    ds = ds.batch(batch_size)
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
```
<br></br>
> CIFAR - 10 의 라벨 갯수 확인
```python
ds_info.features["label"].num_classes
```
<br></br>
> CIFAR - 10 의 라벨 (클래스) 명 확인
```python
ds_info.features["label"].names
```
<br></br>
> CIFAR - 10 의 학습 데이터 확인
```python
fig = tfds.show_examples(ds_train, ds_info)
```
<br></br>
> CIFAR - 10 의 테스트 데이터 확인
```python
fig = tfds.show_examples(ds_test, ds_info)
```
<br></br>


## Ablation Study 실습 (2) 블록 구성하기

논문 모델을 그대로 구현하는 것 만큼 구현시 반복되는 부분을 줄여 하이퍼파라미터 및 변수를 변경하는 것 역시 중요하다.

이를 위해서는 모델 구조가 변경될 때 쉽게 바꿀 수 있게

딥러닝 모델에서 주요 구조를 모듈화 시켜 바꿔 쓸 수 있는 단위를 **블록 (block)** 이라 한다.

즉, 블록은 레이어 (layer) 가 모여 만들어진 단위이며, 이 블록을 만들 수 있어야 한다.

ResNet 은 ResNet - 18, 34, 50, 101, 152 5개의 네트워크가 존재한다.

<br></br>
### VGG 기본 블록 만들기

ResNet 블록을 구현하기에 앞서 비교적 간단한 VGG 를 구현해보자.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/vgg_structure.max-800x600.png)

VGG 역시 VGG - 16, 19 등 여러 버전이 존재한다.

VGG 모델은 몇 개의 CNN 레이어와 Max pooling 레이어 1개로 이루어진다. 즉, 1개의 블록에 CNN 레이어들과 Max pooling 레이어를 필수로 가진다.

CNN 은 모두 3 x 3 크기의 커널을 가지며 블록 간 CNN 레이어의 채널은 하나로 유지되지만 서로 다른 블록 간 CNN 레이어의 채널 수는 다를 수 있다.

> VGG 블록 생성
```python
# function for building VGG Block

def build_vgg_block(input_layer,
                    num_cnn=3, 
                    channel=64,
                    block_num=1,
                   ):
    # 입력 레이어
    x = input_layer

    # CNN 레이어
    for cnn_num in range(num_cnn):
        x = keras.layers.Conv2D(
            filters=channel,
            kernel_size=(3,3),
            activation='relu',
            kernel_initializer='he_normal',
            padding='same',
            name=f'block{block_num}_conv{cnn_num}'
        )(x)    

    # Max Pooling 레이어
    x = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        name=f'block{block_num}_pooling'
    )(x)

    return x
```
`blocknum` 은 레이어의 이름을 붙여주기 위함이며, `input_shape` 는 summary 를 출력하기 위함이다.
<br></br>
> VGG 블록의 입력 레이어와 출력 레이어 설정
```python
vgg_input_layer = keras.layers.Input(shape=(32,32,3))   # 입력 레이어 생성
vgg_block_output = build_vgg_block(vgg_input_layer)    # VGG 블록 생성
```
keras 에서는 `input_layer` 메서드가 있으며, 이를 통해 블록을 추가할 수 있다. 

`build_vgg_block()` 를 통해 블록 레이어를 생성하고 출력값을 얻을 수 있다.

케라스의 `Model` 클래스에서 `input`과 `output`을 정의해주면 간단히 블록의 모델을 확인이 가능하다.

+ 참고 : [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
<br></br>
> 블록 1개짜리 model 생성
```python
# 블록 1개짜리 model 생성
model = keras.Model(inputs=vgg_input_layer, outputs=vgg_block_output)  

model.summary()
```
<br></br>


## Ablation Sutdy 실습 (3) VGG Complete Model

기본 블록을 만드는 과정을 참고하여 전체 VGG 모델을 만들어보자.

<br></br>
### VGG - 16 

VGG - 16 과 19 블록 별 CNN 레이어와 채널 수가 다르기 때문에 리스트를 통해 CNN 의 수와 채널을 전달한다.

>VGG 모델 자체를 생성하는 함수
```python
# VGG 모델 자체를 생성하는 함수입니다.
def build_vgg(input_shape=(32,32,3),
              num_cnn_list=[2,2,3,3,3],
              channel_list=[64,128,256,512,512],
              num_classes=10):
    
    assert len(num_cnn_list) == len(channel_list) #모델을 만들기 전에 config list들이 같은 길이인지 확인합니다.
    
    input_layer = keras.layers.Input(shape=input_shape)  # input layer를 만들어둡니다.
    output = input_layer
    
    # config list들의 길이만큼 반복해서 블록을 생성합니다.
    for i, (num_cnn, channel) in enumerate(zip(num_cnn_list, channel_list)):
        output = build_vgg_block(
            output,
            num_cnn=num_cnn, 
            channel=channel,
            block_num=i
        )
        
    output = keras.layers.Flatten(name='flatten')(output)
    output = keras.layers.Dense(4096, activation='relu', name='fc1')(output)
    output = keras.layers.Dense(4096, activation='relu', name='fc2')(output)
    output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(output)
    
    model = keras.Model(
        inputs=input_layer, 
        outputs=output
    )
    return model
```
<br></br>
> 기본값을 통한 VGG - 16 모델 생성
```python
# 기본값을 그대로 사용해서 VGG 모델을 만들면 VGG-16이 됩니다.
vgg_16 = build_vgg()

vgg_16.summary()
```
<br></br>

### VGG - 19

VGG  -16 의 구성 (configuration) 을 변경해 VGG - 19 모델을 만들 수 있다.

> VGG - 19 모델 생성
```python
# 원하는 블록의 설계에 따라 매개변수로 리스트를 전달해 줍니다.
vgg_19 = build_vgg(
    num_cnn_list=[2,2,4,4,4],
    channel_list=[64,128,256,512,512]
)

vgg_19.summary()
```
<br></br>


## Ablation Study 실습 (4) VGG - 16 vs VGG - 19

VGG - 16 과 VGG - 19 의 성능을 비교해보자.

>CIFAR - 10 데이터셋 불러오기
```python
BATCH_SIZE = 256
EPOCH = 2

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)
ds_train = apply_normalize_on_dataset(ds_train, batch_size=BATCH_SIZE)
ds_test = apply_normalize_on_dataset(ds_test, batch_size=BATCH_SIZE)
```
<br></br>
> VGG - 16 모델 생성 및 학습
```python
vgg_16.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01, clipnorm=1.),
    metrics=['accuracy'],
)

history_16 = vgg_16.fit(
    ds_train,
    steps_per_epoch=int(ds_info.splits['train'].num_examples/BATCH_SIZE),
    validation_steps=int(ds_info.splits['test'].num_examples/BATCH_SIZE),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)
```
BATCH_SIZE 가 커지면 학습 시간이 소폭 감소하며, 40 epoch 이상을 권장한다. 하지만 20 epoch 로도 어느 정도의 성능을 얻을 수 있다.
<br></br>
>VGG - 19 모델 생성 및 학습
```python
vgg_19.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01, clipnorm=1.),
    metrics=['accuracy'],
)

history_19 = vgg_19.fit(
    ds_train,
    steps_per_epoch=int(ds_info.splits['train'].num_examples/BATCH_SIZE),
    validation_steps=int(ds_info.splits['test'].num_examples/BATCH_SIZE),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)
```
<br></br>
> VGG - 16, VGG - 19 의 훈련과정을 시각화
```python
import matplotlib.pyplot as plt

plt.plot(history_16.history['loss'], 'r')
plt.plot(history_19.history['loss'], 'b')
plt.title('Model training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['vgg_16', 'vgg_19'], loc='upper left')
plt.show()
```
<br></br>
> 두 모델의 검증 정확도 (validation accuracy) 비교
```python
plt.plot(history_16.history['val_accuracy'], 'r')
plt.plot(history_19.history['val_accuracy'], 'b')
plt.title('Model validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['vgg_16', 'vgg_19'], loc='upper left')
plt.show()
```
VGG - 19 가 조금 더 높은 정확도를 보임을 알 수 있다.
<br></br>
