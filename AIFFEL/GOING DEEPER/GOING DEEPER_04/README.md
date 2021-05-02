# 04. 이미지 어디까지 우려볼까?

## 학습 목표

1.  Augmentation을 모델 학습에 적용하기
2.  Augmentation의 적용을 통한 학습 효과 확인하기
3.  최신 data augmentation 기법 구현 및 활용하기


## 학습 내용

1.  Augmentation 적용 (1) 데이터 불러오기
2.  Augmentation 적용 (2) Augmentation 적용하기
3.  Augmentation 적용 (3) 비교 실험하기
4.  심화 기법 (1) Cutmix Augmentation
5.  심화 기법 (2) Mixup Augmentation
6.  프로젝트: CutMix 또는 Mixup 비교실험하기

<br></br>
> 작업디렉토리 구성
```bash
$ mkdir -p ~/aiffel/data_augmentation/data
```
<br></br>

## Augmentation 적용 (01). 데이터 불러오기

Augmentation 을 텐서플로우 모델 학습에 어떻게 적용할 수 있는지 알아보자.

지금까지는 모델 학습 전 데이터를 전처리해 입력값으로 사용해 온 것 처럼 Augmentation 도 입력 이미지 데이터를 변경해주는 과정으로 일반적인 이미지 데이터 전처리 방법과 동일하다.
<br></br>
> 필요 라이브러리 임포트
```python
pip install tensorflow_datasets

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
```
<br></br>
> GPU 환경 확인
```python
tf.config.list_physical_devices('GPU')
print(tf.__version__)
```
<br></br>

Augmentation 적용을 실습을 위해 `stanford_dog` 데이터셋을 사용하자.

+ 참고 : [stanford_dogs of Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/stanford_dogs)
<br></br>
> 데이터 다운로드
```python
import urllib3
urllib3.disable_warnings()
(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)
```
데이터셋은 프로젝트 디렉토리가 아닌 Tensorflow Datasets 기본 디렉토리에 저장한다.
<br></br>
> 데이터 확인
```python
fig = tfds.show_examples(ds_train, ds_info)
```
<br></br>

## Augmentation 적용 (02). Augmentation 적용하기

### 텐서플로우 Random Augmentation API 사용하기

augmentation 의 기법은 무수히 많다. 텐서플로우에서는 많은 augmentation 기법을 API 를 통해 적용할 수 있도록 제공하고 있다.

이미지셋에 대해 랜덤 확률로 바로 적용할 수 있는 augmentation 함수는 다음과 같다.

-   `random_brightness()`
-   `random_contrast()`
-   `random_crop()`
-   `random_flip_left_right()`
-   `random_flip_up_down()`
-   `random_hue()`
-   `random_jpeg_quality()`
-   `random_saturation()`

<br></br>
> Augmentation 을 위한 기본적인 전처리 함수 생성
```python
def normalize_and_resize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255., label
```
이미지를 변환하는 전처리 함수는 다음과 같은 형태를 가지게 된다.
```python
def 전처리_함수(image, label):  # 변환할 이미지와 
	# 이미지 변환 로직 적용 
	new_image = 이미지_변환(image) 
	return new_image, label
```
<br></br>

이를 통해 이미지 변환의 결과를 리턴받고, 리턴받은 이미지를 다음 전처리 함수의 입력으로 사용할 수 있게 구조화하는 것이다.

위 함수는 입력받은 이미지를 0 ~ 1 사이의 float 32 로 normalize 하고, (224, 224) 사이즈로 resize 한다.

이후 훈련 및 테스트용으로 사용될 모든 이미지에 적용한다.

> `random_flip_left_right()` 과 `random_brightness()` random augmentationwjrdyd
```python
def augment(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image,label
```
"flip" 의 경우 좌우 대칭을 수행하며, "brightness" 를 조절하여 다양한 환경에서 얻어진 이미지에 대응할 수 있도록 하였다.
<br></br>

이후 Augmentation 을 통해 얻게된 다양한 형태의 가공된 데이터셋을 구현하기 위한 메인 함수를 `apply_normalize_on_dataset()` 로 정의한다.

일반적인 전처리 과정 (normalize, resize, augmenation, shuffle) 을 적용한다.

주의할 점은 shuffle이나 augmentation은 테스트 데이터셋에는 적용하지 않아야 한다는 점이다.

+ 여러 결과를 조합하기 위한 앙상블 (ensemble) 방법 중 하나로 테스트 데이터셋에 augmentation 을 적용하는 test-time augmentation 이라는 방법이 있다. 캐글 등의 경쟁 머신러닝에 많이 사용된다.
	+ 참고 : [test - time augmentation](https://hwiyong.tistory.com/215)

<br></br>
 또한 비교실험을 위해 `with_aug` 매개변수를 통해 augmentation 적용 여부를 결정하도록 한다.

+ 참고 : [f.data.Datasets.map()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map)
<br></br>
> 데이터셋 가공을 위한 메인함수 `apply_normalize_on_dataset()` 생성
```python
# 데이터셋(ds)을 가공하는 메인함수
def apply_normalize_on_dataset(ds, is_test=False, batch_size=16, with_aug=False):
    ds = ds.map(
        normalize_and_resize_img,  # 기본적인 전처리 함수 적용
        num_parallel_calls=2
    )
    if not is_test and with_aug:
        ds = ds.map(
            augment,       # augment 함수 적용
            num_parallel_calls=2
        )
    ds = ds.batch(batch_size)
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
```
<br></br>

### Random Augmentation 직접 구현하기

> `tf.image` 의 함수를 이용해 augmentation 을 랜덤하게 적용하는 `augment2()` 함수 생성
```python
# make random augment function
def augment2(image,label):
    image = tf.image.central_crop(image, np.random.uniform(0.50, 1.00)) # 50%의 확률로 이미지 가운데 부분을 crop합니다.
    image = tf.image.resize(image, INPUT_SHAPE) # crop한 이미지를 원본 사이즈로 resize
    return image, label
```
<br></br>

## Augmentation 적용 (03). 비교실험하기

이제 augmentation 을 적용한 데이터로 학습한 모델과 적용하지 않은 데이터로 학습한 모델의 성능을 비교해보자.

> Augmentation 을 적용하지 않은 데이터셋으로 학습시킬 모델 불러오기
> (케라스의 `ResNet50` 중에서 `imagenet` 에 훈련된 모델을 불러오기)
```python
num_classes = ds_info.features["label"].num_classes
resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```
`include_top`은 마지막 fully connected layer 를 포함할지 여부를 나타내며, 해당 레이어를 포함하지 않고 생성하면 특성 추출기 (feature extractor) 부분만 불러와 우리의 필요에 맞게 수정된 fully connected layer 를 붙여서 활용할 수 있다.

이미지넷 (ImageNet) 과 우리의 테스트셋이 서로 다른 클래스를 가지므로, 마지막에 추가해야 하는 fully connected layer 의 구조(뉴런의 개수) 또한 다르기 때문에 이와 같이 사용한다.
<br></br>
> Augmentation 을 적용한 데이터셋으로 학습시킬 `ResNet` 모델을 하나 더 생성
```python
aug_resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```
<br></br>
> augmentation 적용한 데이터셋과 적용하지 않은 데이터셋 분리
```python
(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)
ds_train_no_aug = apply_normalize_on_dataset(ds_train, with_aug=False)
ds_train_aug = apply_normalize_on_dataset(ds_train, with_aug=True)
ds_test = apply_normalize_on_dataset(ds_test, is_test = True)
```
데이터셋에 `apply_normalize_on_dataset()`에서 `with_aug` 를 `False` 로 주어 augmentation 이 적용되지 않도록 하고, 다른 하나는 `True` 로 주어 augmentation 을 적용한다.
<br></br>
> 2개의 모델에 augmentation 적용 여부에 따른 데이터셋을 학습시키고 검증 진행
```python
#EPOCH = 20  # Augentation 적용 효과를 확인하기 위해 필요한 epoch 수
EPOCH = 3

tf.random.set_seed(2020)
resnet50.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01),
    metrics=['accuracy'],
)

aug_resnet50.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01),
    metrics=['accuracy'],
)

history_resnet50_no_aug = resnet50.fit(
    ds_train_no_aug, # augmentation 적용하지 않은 데이터셋 사용
    steps_per_epoch=int(ds_info.splits['train'].num_examples/16),
    validation_steps=int(ds_info.splits['test'].num_examples/16),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)

history_resnet50_aug = aug_resnet50.fit(
    ds_train_aug, # augmentation 적용한 데이터셋 사용
    steps_per_epoch=int(ds_info.splits['train'].num_examples/16),
    validation_steps=int(ds_info.splits['test'].num_examples/16),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)
```
<br></br>
> 모델 훈련과정 시각화
```python
plt.plot(history_resnet50_no_aug.history['val_accuracy'], 'r')
plt.plot(history_resnet50_aug.history['val_accuracy'], 'b')
plt.title('Model validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['No Augmentation', 'With Augmentation'], loc='upper left')
plt.show()
```
<br></br>
> 모델 훈련과정을 확대해서 시각화
```python
plt.plot(history_resnet50_no_aug.history['val_accuracy'], 'r')
plt.plot(history_resnet50_aug.history['val_accuracy'], 'b')
plt.title('Model validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['No Augmentation', 'With Augmentation'], loc='upper left')
plt.grid(True)
plt.ylim(0.72, 0.76)
plt.show()
```
augmentation 을 적용한 경우 학습시간이 더 오래 걸리지만 accuracy 가 더 높게 형성됨을 알 수 있다.
<br></br>

## 심화 기법 (01). Cutmix Augmentation

Augmentation 의 보다 복잡한 방법을 알아보자.

첫 번째로, CutMix Augmentation 방법이다.

CutMix 는 네이버 클로바에서 발표한 방법으로 이미지를 자르고 섞는 것이다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-2-P-2.max-800x600.png)
<br></br>
일반적으로 사용해왔던 방식은 위 그림과 같다.

Mixup 은 특정 비율로 픽셀별 값을 섞는 방법이며, Cutout 은 이미지를 잘라내는 방식이다.

CutMix 는 이 둘을 혼합한 방식으로 일정 영역을 잘라 붙여주는 방법이다.

CutMix 는 이미지를 섞는 부분과 섞은 이미지에 맞게 라벨을 섞는 것 까지 포함된다.

+ 참고 : 
	-   [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899.pdf)
	-   [Chris Deotte's CutMix and MixUp on GPU/TPU](https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu)

<br></br>
### (01). 이미지 섞기

먼저 두 개의 이미지를 섞어보자.

배치 내의 이미지를 두 개 골라 섞는데, 이미지에서 잘라서 섞어주는 영역을 바운딩 박스라 부른다.
<br></br>
> 바운딩 박스의 위치를 랜덤하게 뽑아 이를 잘라내고 두 개의 이미지를 섞어주는 함수를 생성
> (이미지를 텐서로 만들고 텐서플로우 연산을 사용하며, 이때 이미지는 `tfds` 에서 한 장을 뽑아 사용)
```python
import matplotlib.pyplot as plt

# 데이터셋에서 이미지 2개를 가져옵니다. 
for i, (image, label) in enumerate(ds_train_no_aug.take(1)):
    if i == 0:
        image_a = image[0]
        image_b = image[1]
        label_a = label[0]
        label_b = label[1]
        break

plt.subplot(1,2,1)
plt.imshow(image_a)

plt.subplot(1,2,2)
plt.imshow(image_b)
```
<br></br>
> 첫번째 이미지 a를 바탕 이미지로 하고 거기에 삽입할 두번째 이미지 b가 있을 때, a에 삽입될 영역의 바운딩 박스의 위치를 결정하는 함수를 생성
```python
def get_clip_box(image_a, image_b):
    # image.shape = (height, width, channel)
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]
    
    # get center of box
    x = tf.cast( tf.random.uniform([],0, image_size_x),tf.int32)
    y = tf.cast( tf.random.uniform([],0, image_size_y),tf.int32)

    # get width, height of box
    width = tf.cast(image_size_x * tf.math.sqrt(1-tf.random.uniform([],0,1)),tf.int32)
    height = tf.cast(image_size_y * tf.math.sqrt(1-tf.random.uniform([],0,1)),tf.int32)
    
    # clip box in image and get minmax bbox
    xa = tf.math.maximum(0, x-width//2)
    ya = tf.math.maximum(0, y-height//2)
    xb = tf.math.minimum(image_size_x, x+width//2)
    yb = tf.math.minimum(image_size_y, y+width//2)
    
    return xa, ya, xb, yb

xa, ya, xb, yb = get_clip_box(image_a, image_b)
print(xa, ya, xb, yb)
```
이미지 a, b가 모두 (224, 224) 로 resize 되어 두 이미지의 width, height 가 같은 경우로 가정할 수 있지만, CutMix 공식 repo 에서는 width, height 가 다르더라도 가변적으로 적용할 수 있도록 구현되어 있기 때문에, 임의의 이미지 사이즈에 대해서도 유연하게 대응 가능하도록 구현하였다.
<br></br>
> 바탕이미지 a에서 바운딩 박스 바깥쪽 영역을, 다른 이미지 b에서 바운딩 박스 안쪽 영역을 가져와서 합치는 함수를 구현
```python
# mix two images
def mix_2_images(image_a, image_b, xa, ya, xb, yb):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0] 
    one = image_a[ya:yb,0:xa,:]
    two = image_b[ya:yb,xa:xb,:]
    three = image_a[ya:yb,xb:image_size_x,:]
    middle = tf.concat([one,two,three],axis=1)
    top = image_a[0:ya,:,:]
    bottom = image_a[yb:image_size_y,:,:]
    mixed_img = tf.concat([top, middle, bottom],axis=0)
    
    return mixed_img

mixed_img = mix_2_images(image_a, image_b, xa, ya, xb, yb)
plt.imshow(mixed_img.numpy())
```
<br></br>

### (02). 라벨 섞기

이미지를 섞었으면 라벨도 함께 섞어줘야 한다.

CutMix 는 면적에 비례해서 라벨을 섞어주기 때문에 섞인 이미지의 전체 이미지 대비 비율을 계산해 두 가지 라벨의 비율을 더해준다.

***
예 : A 클래스를 가진 원래 이미지 `image_a` 와 B 클래스를 가진 이미지 `image_b` 를 섞을 때 `image_a` 를 0.4만큼 섞었을 경우, 0.4만큼의 클래스 A, 0.6만큼의 클래스 B를 가지도록 해야한다.

이때 라벨 벡터는 보통 클래스를 표시하듯 클래스 1개만 1의 값을 가지는 원-핫 인코딩이 아니라 A와 B 클래스에 해당하는 인덱스에 각각 0.4, 0.6을 배분하는 방식을 사용한다.
***
<br></br>
> 섞인 두 이미지에 대해 라벨을 만들 때 적절한 비율로 라벨을 합쳐주는 함수를 구현
```python
# mix two labels
def mix_2_label(label_a, label_b, xa, ya, xb, yb, num_classes=120):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0] 
    mixed_area = (xb-xa)*(yb-ya)
    total_area = image_size_x*image_size_y
    a = tf.cast(mixed_area/total_area, tf.float32)

    if len(label_a.shape)==0:
        label_a = tf.one_hot(label_a, num_classes)
    if len(label_b.shape)==0:
        label_b = tf.one_hot(label_b, num_classes)
    mixed_label = (1-a)*label_a + a*label_b
    return mixed_label

mixed_label = mix_2_label(label_a, label_b, xa, ya, xb, yb)
mixed_label
```
<br></br>
> `mix_2_images()` 와 `mix_2_label()` 을 활용하여 배치 단위의 `cutmix()` 함수를 구현
```python
def cutmix(image, label, prob = 1.0, batch_size=16, img_size=224, num_classes=120):
    mixed_imgs = []
    mixed_labels = []

    for i in range(batch_size):
        image_a = image[i]
        label_a = label[i]
        j = tf.cast(tf.random.uniform([],0, batch_size),tf.int32)
        image_b = image[j]
        label_b = label[j]
        xa, ya, xb, yb = get_clip_box(image_a, image_b)
        mixed_imgs.append(mix_2_images(image_a, image_b, xa, ya, xb, yb))
        mixed_labels.append(mix_2_label(label_a, label_b, xa, ya, xb, yb))

    mixed_imgs = tf.reshape(tf.stack(mixed_imgs),(batch_size, img_size, img_size, 3))
    mixed_labels = tf.reshape(tf.stack(mixed_labels),(batch_size, num_classes))
    return mixed_imgs, mixed_label
```
<br></br>

## 심화기법 (02). Mixup Augmentation

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-2-P-3.max-800x600.png)
<br></br>
Mixup 은 CutMix 보다 간단하게 이미지와 라벨을 섞어준다.

Mixup 은 두 개의 이미지의 픽셀별 값을 비율에 따라 섞어주는 방식을 사용한다.
<br></br>
+ 참고 : -   [논문 - mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
<br></br>
> Mixup 을 수행하는 함수 구현
```python
# function for mixup
def mixup_2_images(image_a, image_b, label_a, label_b):
    a = tf.random.uniform([],0,1)
    
    if len(label_a.shape)==0:
        label_a = tf.one_hot(label_a, num_classes)
    if len(label_b.shape)==0:
        label_b = tf.one_hot(label_b, num_classes)
    mixed_image= (1-a)*image_a + a*image_b
    mixed_label = (1-a)*label_a + a*label_b
    
    return mixed_image, mixed_label

mixed_img, _ = mixup_2_images(image_a, image_b, label_a, label_b)
plt.imshow(mixed_img.numpy())
print(mixed_label)
```
<br></br>
> Mixup 을 수행하는 함수를 활용하여 배치 단위의 mixup() 함수를 구현
```python
def mixup(image, label, prob = 1.0, batch_size=16, img_size=224, num_classes=120):
    mixed_imgs = []
    mixed_labels = []

    for i in range(batch_size):
        image_a = image[i]
        label_a = label[i]
        j = tf.cast(tf.random.uniform([],0, batch_size),tf.int32)
        image_b = image[j]
        label_b = label[j]
        mixed_img, mixed_label = mixup_2_images(image_a, image_b, label_a, label_b)
        mixed_imgs.append(mixed_img)
        mixed_labels.append(mixed_label)

    mixed_imgs = tf.reshape(tf.stack(mixed_imgs),(batch_size, img_size, img_size, 3))
    mixed_labels = tf.reshape(tf.stack(mixed_labels),(batch_size, num_classes))
    return mixed_imgs, mixed_labels
```
<br></br>
