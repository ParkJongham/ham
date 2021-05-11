# 14. 폐렴아 기다려라!

딥러닝 기술이 산업적으로 명확한 용도를 입증한 도메인 중 하나는 의료인공지능 분야이다. 의료영상을 분석하는 일은 전문적인 훈련을 받은 숙련된 의료인력만 가능했지만, 딥러닝 기술의 적용으로 숙련자 수준 이상의 정확도를 바탕으로 영상분석 인력의 개인적 편차, 주관적 판단, 피로에 의한 오진 등의 부정확성을 극복할 수 있는 좋은 대안이 되고있다.

의료영상의 특징은 다음과 같다.

-   의료영상 이미지는 개인정보 보호 등의 이슈로 인해 데이터를 구하는 것이 쉽지 않다.
-   라벨링 작업 자체가 전문적 지식을 요하므로 데이터셋 구축 비용이 비싸다.
-   희귀질병을 다루는 경우 데이터를 입수하는 것 자체가 드문 일이다.
-   음성 / 양성 데이터간 불균형이 심해 학습에 주의가 필요하다.
-   이미지만으로 진단이 쉽지 않아 다른 데이터와 결합해서 해석해야 할수도 있다.

때문에 딥러닝 영상처리 기술 외에 의료 도메인 지식 및 의료영상에 대한 이해가 필요하다.
<br></br>

## 의료 영상에 대해

### 사람 속을 보는 방법

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/01.png)
<br></br>
만약 폐를 진단한다면 위 그림과 같이 X - RAY 촬영 및 CT 영상을 찍어 봄으로써 확인할 수 있을 것이다.

<br></br>
### 의료 영상 종류

#### X - RAY

![](https://aiffelstaticprd.blob.core.windows.net/media/images/02.max-800x600.png)
<br></br>
X - RAY 는 전자를 물체에 충돌시킴으로써 발생하는 투과력이 강한 복사선 (전자기파) 이다.

방사선의 일종으로 지방, 근육 등과 같이 밀도가 낮은 물질은 통과하지만 뼈와 같은 밀도가 높은 물질은 잘 통과하지 못한다.
<br></br>

#### CT 

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/04.png)
<br></br>
CT (Computed Tomography) 는 환자를 중심으로 X - RAY 를 빠르게 회전시켜 3D 이미지를 만들어내는 것이다.

3 차원 이미지를 형성하며, 기본 구조 및 종양, 이상을 쉽게 식별할 수 있고 위치를 파악할 수 있다.

CT 의 단층, 즉 신체의 단면 이미지를 Slice 라고 하며, 이 Slice 는 X - RAY 보다 더 자세한 정보를 포함한다.
<br></br>

#### MRI

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/05.png)
<br></br>
MRI (Magnetic Resonance Imaging) 은 자기 공명 영상이라고도 불리며, 신체의 해부학적 과정과 생리적 과정을 보기 위해 사용한다.

MRI 스캐너는 강한 자기장을 통해 신체 기관의 이미지를 생성하며, CT, X - RAY 와 달리 방사선을 사용하지 않는다.

+ 참고 : [유투브 : 의료영상기기의 원리](https://youtu.be/J_Owz3YBkD0)
<br></br>

## X - RAY 이미지

### 의료영상 자세 분류

![](https://aiffelstaticprd.blob.core.windows.net/media/images/06.max-800x600.png)
<br></br>

위 그림은 의료 영상 촬영의 방향을 나타낸것이다. 촬영은 Sagittal plane, Coronal plane, Transverse plane, 3 가지 방향의 단면으로 나눠 진행된다.

Sagittal plane 는 시상면으로 사람의 왼쪽과 오른쪽을 나누는 면을 의미하며, Coronal plane 은 관상면으로 사람의 앞, 뒤를 나누는 면이다. Transverse plane 는 횡단면, 수평면을 의미하며 사람의 상, 하로 나누는 면을 의미한다.

오늘 우리가 사용할 데이터는 관상면 이미지로 구성되어 있다.
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/08.max-800x600.png)
<br></br>
위 기름은 인체의 해부학적 위치를 나타낸 그림이다.

오른쪽 이미지에서 왼쪽 얼굴에 오른쪽 이라고 표기되어 있는 것을 볼 수 있다. 즉, 영상 이미지를 볼 때는 보고자 하는 방향의 반대 방향을 보는 것이다.
<br></br>

### X - RAY 의 특성

![](https://aiffelstaticprd.blob.core.windows.net/media/images/09.max-800x600.png)
<br></br>
X - RAY 는 전자기파를 몸에 투과시켜 이미지화 시킨 것으로, 흑백 명암으로 나오게 된다.

위 그림은 손을 찍은 사진으로, 밀도가 낮은 지방과 근육은 비교적 쉽게 투과되어 비교적 어둡게, 밀도가 높은 뼈 부분은 투과되지 못해 밝게 나온다.

따라서 뼈는 흰색으로 나타나며, 근육 및 지방은 연한 회색, 공기는 검은색으로 나타난다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/10.png)
<br></br>
위 그림은 폐를 찍은 사진으로, 갈비뼈는 흰색, 폐는 검은색, 어깨 쪽의 지방 및 근육은 연한회색으로 나타난다.

폐는 공기로 차있기 때문에 검은색으로 나오게된다.
<br></br>

## 폐렴을 진단해 보자 (01)

우리는 의료 인공지능을 활용하여 폐렴 (Pneumonia) 를 진단하는 모델을 만들어 보고자 한다.

데이터로는 캐글의 Chest X - RAY Images 를 활용한다.
해당 이미지는 중국 광저우 여성 및 어린이 병원의 1 ~ 5 세 소아 환자의 흉부 X - RAY 영상이다.

+ [데이터 : Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
<br></br>
> 데이터셋 다운로드 및 작업 디렉토리 설정
``` bash
$ mkdir -p ~/aiffel/chest_xray 
$ cd ~/Downloads && unzip archive.zip -d ~/aiffel
```
<br></br>

데이터는 train, test, val 폴더로 구성되어 있으며, 각 폴더에 폐렴 과 정상 에 대한 하위 폴더를 가지고 있다.

총 5,856 개의 X - RAY JPEG 이미지와 2 개의 범주 (폐렴 / 정상) 이 있다.

폐렴은 폐에 염증이 생기는 것으로, 구별법은 생각보다 간단하다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/11_l7pucvb.png)
<br></br>
위 그림은 폐렴 X - RAY 영상이며, 폐렴일 경우 사진상 양상의 음영 (폐 부위에 희미한 그림자) 증가가 관찰된다.

간단한 구별법이지만, 실제오 영상을 봤을 때 실제 폐렴으로 인해 발생한 희미함인지, 다른 이유로 인한 것인지 정확히 구분하기 힘들다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/12_e0QcAmT.png)
<br></br>
위 그림은 정상적인 폐, 세균성 폐렴, 바이러스성 폐렴을 나타낸 것이다.

세균성 폐렴은 일반적으로 오른쪽 상부 엽 (흰색 화살표) 에 나타나는 반면 바이러스성 폐렴은 양쪽 폐에서 보다 확산된 Interstitial (조직 사이에 있는) 패턴으로 나타난다.

폐렴은 이러한 패턴을 볼 수 있으며, 이러한 패턴을 잘 읽어내는 딥러닝 알고리즘을 통해 학습시키게 된다.
<br></br>
### Set - up

> 사용할 라이브러리 가져오기
```python
import re    # 정규표현식 관련된 작업에 필요한 패키지
import os    # I/O 관련된 작업에 필요한 패키지 
import pandas as pd     # 데이터 전처리 관련된 작업에 필요한 패키지
import numpy as np      # 데이터 array 작업에 필요한 패키지
import tensorflow as tf  # 딥러닝 관련된 작업에 필요한 패키지
import matplotlib.pyplot as plt    # 데이터 시각화에 관련된 작업에 필요한 패키지
from sklearn.model_selection import train_test_split  # 데이터 전처리에 필요한 패키지
```
<br></br>
> 앞으로 필요한 변수 생성
```python
# 데이터 로드할 때 빠르게 로드할 수 있도록하는 설정 변수
AUTOTUNE = tf.data.experimental.AUTOTUNE

# 데이터 ROOT 경로 변수
ROOT_PATH = os.path.join(os.getenv('HOME'))

# BATCH_SIZE 변수
BATCH_SIZE = 16

# X-RAY 이미지 사이즈 변수
IMAGE_SIZE = [180, 180]

# EPOCH 크기 변수
EPOCHS = 25

print(ROOT_PATH)
```
<br></br>
> 데이터 가져오기 (데이터 갯수 파악)
```python
train_filenames = tf.io.gfile.glob(str(ROOT_PATH + '/chest_xray/train/*/*'))
test_filenames = tf.io.gfile.glob(str(ROOT_PATH + '/chest_xray/test/*/*'))
val_filenames = tf.io.gfile.glob(str(ROOT_PATH + '/chest_xray/val/*/*'))

print(len(train_filenames))
print(len(test_filenames))
print(len(val_filenames))
```
train 안에는 5216 개, test 안에는 624 개, val 안에는 16 개가 있음을 확인할 수 있다.

비율로는 89%, 10.7%, 0.3% 로 val 데이터 개수가 너무 부족하기 때문에 train 에서 val 쓰일 데이터를 옮기도록 한다.
<br></br>
> train 데이터와 val 데이터를 8 : 2 로 분할
```python
filenames = tf.io.gfile.glob(str(ROOT_PATH + '/chest_xray/train/*/*'))
filenames.extend(tf.io.gfile.glob(str(ROOT_PATH + '/chest_xray/val/*/*')))

# train, test(val) dataset으로 분할. test_size에 0.2는 20%롤 의미함.
train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)

print(len(train_filenames))
print(len(val_filenames))
```
train은 4185 개, test는 624 개, val은 1047 개 로 나눠주었다.
<br></br>
> train 데이터의 정상 이미지 수와 폐렴 이미지 수 확인
```python
COUNT_NORMAL = len([filename for filename in train_filenames if "NORMAL" in filename])
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len([filename for filename in train_filenames if "PNEUMONIA" in filename])
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))
```
정상 이미지 보다 폐렴 이미지 수가 3 배 더 많이 있음을 알 수 있다.

보통 이미지 분류를 수행하는 CNN 모델의 경우 클래스별 데이터가 균형적일 수록 학습을 더 잘 한다.

따라서 이후 데이터의 불균형을 처리해줘야한다.

test, val 데이터셋은 평가를 위해서만 사용되므로, 데이터가 불균형이어도 상관없다.
<br></br>
> tf.data 인스턴스 생성
```python
train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)
```
<br></br>
> Train, Validation 데이터 셋 개수 확인
```python
TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))
```
<br></br>
> 라벨 이름 확인
```python
CLASS_NAMES = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
                        for item in tf.io.gfile.glob(str(ROOT_PATH + "/chest_xray/train/*"))])
print(CLASS_NAMES)
```
정상을 의미하는 NORMAL 라벨과 폐렴을 의미하는 PNEUMONIA 라벨이 존재한다.

제목으로 나눠져있어 현재 라벨 데이터가 없으므로 라벨 데이터를 만들어 줘야한다.
<br></br>
> 라벨 데이터를 만들어주는 함수 생성
```python
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == "PNEUMONIA"   # 폐렴이면 양성(True), 노말이면 음성(False)를 리턴하게 합니다.
```
<br></br>
> 이미지 사이즈 통일 및 사이즈 축소를 위한 함수 생성
```python
def decode_img(img):
  # 이미지를 uint8 tensor로 바꾼다.
  img = tf.image.decode_jpeg(img, channels=3)
  # img를 범위 [0,1]의 float32 데이터 타입으로 바꾼다.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # img의 이미지 사이즈를 IMAGE_SIZE에서 지정한 사이즈로 수정한다.
  return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label
```
process_path 함수에서 decode_img 함수를 이용해서 이미지의 데이터 타입을 float 으로 바꾸고 사이즈를 변경하며, get_label 을 이용하여 라벨 값을 가져온다.
<br></br>
> train, validation 데이터셋 생성
> (num_parallel_calls 파라미터에서 set - up 에서 초기화 한 AUTOTUNE 을 활용)
```python
train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
```
<br></br>
> 이미지 리사이즈 및 라벨 확인
```python
for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
```
`train_ds.take(1)`은 하나의 데이터만 가져온다 라는 의미이다.
<br></br>
> test 데이터 셋 생성 및 데이터 갯수도 확인
```python
test_list_ds = tf.data.Dataset.list_files(str(ROOT_PATH + '/chest_xray/test/*/*'))
TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)

print(TEST_IMAGE_COUNT)
```
Tensorflow 에서는 tf.data 파이프라인을 사용해서 학습 데이터를 효율적으로 사용할 수 있도록 해 준다.
<br></br>
> 모델 학습 전에 데이터 효율적으로 사용하도록 준비
```python
def prepare_for_training(ds, shuffle_buffer_size=1000):

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)
```
prepare_for_training() 함수는 학습 데이터를 효율적으로 할 수 있도록 데이터를 변환 시켜주며, shuffle() 을 사용하며 고정 크기 버퍼를 유지하고 해당 버퍼에서 무작위로 균일하게 다음 요소를 선택한다.

repeat() 를 사용하면 epoch를 진행하면서 여러번 데이터셋을 불러오게 되는데, 이때 repeat() 를 사용한 데이터셋의 경우 여러번 데이터셋을 사용할 수 있게 해준다.

예를 들어, 100개의 데이터를 10번 반복하면 1000개의 데이터가 필요한데, repeat() 를 사용하면 자동으로 데이터를 맞춰준다.

batch() 를 사용하면 BATCH_SIZE 에서 정한 만큼의 배치로 주어지며, prefetch() 를 사용하면 학습데이터를 나눠서 읽어온다.
<br></br>

### 데이터 시각화

> train 의 batch 중 첫 번쨰 배치를 추출하고 이미지와 라벨 데이터셋으로 나누는 함수 생성
```python
image_batch, label_batch = next(iter(train_ds))

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(16):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("PNEUMONIA")
        else:
            plt.title("NORMAL")
        plt.axis("off")

show_batch(image_batch.numpy(), label_batch.numpy())
```
<br></br>

## 폐렴을 진단해보자 (02)

### CNN 모델링

대표적인 딥러닝 CNN 모델을 만들고, 결과를 살펴보자.

+ 참고 : [CNN, Convolutional Neural Network 요약](http://taewan.kim/post/cnn/)
<br></br>

> CNN 모델의 Convolution block 생성
```python
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    
    return block
```
conv_block() 의 구성은 Convolution 을 두번 진행하고 Batch Normalization 을 통해서 Gradient vanishing, Gradient Exploding 을 해결하고 Max Pooling 을 수행
<br></br>
> Dense Block  생성
```python
def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block
```
<br></br>

최종적으로 만들 모델은 CNN 모델의 조금 수정한 모델이다.

위 코드에서 Batch Normalization 과 Dropout 이라는 두가지 regularization 기법이 동시에 사용하는 부분이 일반적인 CNN 모델과 조금 다르다.

일반적으로는 잘 사용되지 않거나 금기시되기도 한다.

+ 참고 :  [논문 : Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf)
<br></br>

위 논문에서는 variance shift 를 억제하는 Batch Normalization 과 이를 유발시키는 Dropout 을 동시에 사용하는 것이 어울리지 않는다고 밝히고 있다.

하지만 두 방법을 같이 쓰는 것이 낫다는 견해가 없는 것은 아니며, 동시에 사용하여 성능 향상을 이룬 사례가 있다.

+ 참고 : [논문 : Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks](https://arxiv.org/pdf/1905.05928.pdf)
<br></br>
> 수정된 모델 모델링
```python
def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model
```
<br></br>

### 데이터 Imbalance 처리

앞서 언급한 데이터 불균형을 처리해 줘야한다.

폐렴의 데이터가 정상 폐의 데이터 보다 많았는데, 이러한 데이터의 불균형을 처리하는 방법으로 `Weight balancing` 이라는 방법을 사용한다.

Weight balancing 은 training set 의 각 데이터에서 loss 를 계산할 때 특정 클래스의 데이터에 더 큰 loss 값을 갖도록 가중치를 부여하는 방법으로 Keras는 model.fit()을 호출할 때 파라미터로 넘기는 class_weight 에 이러한 클래스별 가중치를 세팅할 수 있도록 지원하고 있다.

+ 참고 : [딥러닝에서 클래스 불균형을 다루는 방법](https://3months.tistory.com/414)
<br></br>

> `Weight balancing` 처리
```python
weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0 
weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
```
`weight_for_0` 은 'Normal' 이미지에 사용할 weight를, `weight_for_1` 은 'Pneumonia' 이미지에 사용할 weight 를 세팅한다. 

이 weight 들은 'Normal' 과 'Pneumonia' 전체 데이터 건수에 반비례하도록 설정된다.
<br></br>

### 모델 훈련

> 모델 함수를 선언
```python
with tf.device('/GPU:0'):
    model = build_model()

    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=METRICS
    )
```
build_model() 을 model 에 선언하고, 이미지의 라벨이 두 개밖에 없기 때문에 "binary_cross entropy" loss 를, optimizer 로 'adam' 을 사용한다.

성능 평가를 위한 metrics 으로 'accuracy', 'precision', 'recall' 을 사용한다. 
<br></br>
> 모델 학습
```python
with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_ds,
        validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
        class_weight=class_weight,
    )
```
<br></br>

### 결과 확인

> 학습과정 시각화를 통해 결과 확인
```python
fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
```
<br></br>

> 테스트 데이터를 통한 모델 평가
```python
loss, acc, prec, rec = model.evaluate(test_ds)
```
약 80% 에 달하는 성능을 보이고 있음을 확인할 수 있다.

성능을 개선하기위해 가장 간단한 방법으로는 데이터를 많이 확보하는 것이 있다.

하지만 의료 데이터의 경우 구하기 힘들 뿐더러 구하더라도 데이터가 작은 경우가 많아 Data augmentation 방법을 사용한다.

하지만 Data augmentation 은 각 데이터에 최적화된 방법을 찾기가 어렵고 제약사항이 많아 기본적인 이미지 회전, 가우시안 노이즈 추가 방법 등 을 적용한 Data augmentation 을 많이 사용한다.

또한 사람의 장기는 거의 바뀌지 않기 떄문에 노이즈를 추가하는 방법이 있으며, 이 외에도 GAN 을 통해 Data augmentation 을 시도하는 방법이 있다.
<br></br>
