# 10. 나를 찾아줘 - Class Activation Map 만들기


## 학습 목표

1.  Classification model로부터 CAM을 얻어낼 수 있다.
2.  CAM으로 물체의 위치를 찾을 수 있다.
3.  CAM을 시각화 비교할 수 있다.

## 학습 준비

> 작업 디렉토리 설정
```bash
$ mkdir -p ~/aiffel/class_activation_map
```
<br></br>

## CAM, Grad - CAM 용 모델 준비하기 (01). 데이터셋 준비하기

CAM (Class Activation Map) 은 특성을 추출하는 CNN  네트워크 뒤에 GAP (Global Average Pooling) 와 소프트맥스 레이어 (softmax layer) 가 붙는 형태로 구성되어야 한다는 제약이 있는 반면에 Grad - CAM 은 제약이 없다.

먼저 CAM 을 위한 모델을 먼저 구성해, CAM 을 추출해보고, 이 모델이서 Grad - CAM 을 활용해 시각화 결과물울 추출해보자.

Grad - CAM 모델은 구조에 제약이 없으므로, CAM 에만 모델을 맞춰도 충분하다.

CAM은 클래스에 대한 활성화 정도를 나타낸 지도이다. 따라서 기본적으로 우리의 모델은 분류 (classfication) 를 수행하는 모델이어야 하지만, 이미지 내에서 클래스가 활성화 된 위치를 확인하고 이를 정답과 비교하기 위해서는 위치 정보가 기록된 데이터가 함께 있어야 한다.

`Tensorflow Datasets`의 카탈로그에서 이러한 데이터를 확인할 수 있으며, `Cars196` 데이터셋을 사용하도록 한다.

`Cars196` 데이터은 196 개의 차종을 반펼하는 이미지 분류 데이터셋으로 8,144장의 학습용 데이터셋과 8,041장의 평가용 데이터셋으로 구성되어 있으며, 라벨이 위치정보인 바운딩 박스 정보를 포함하고 있다.

+ -   [데이터 셋 : Cars196 in Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/cars196)
<br></br>
> 필요 라이브러리 임포트
```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

import copy
import cv2
from PIL import Image

tf.config.list_physical_devices('GPU')
```
<br></br>
> 데이터셋 준비
```python
# 최초 수행시에는 다운로드가 진행됩니다. 오래 걸릴 수 있으니 유의해 주세요.  
import urllib3
urllib3.disable_warnings()
(ds_train, ds_test), ds_info = tfds.load(
    'cars196',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)
```
<br></br>
> 각 이미지의 클래스와 인덱스 확인
```python
tfds.show_examples(ds_train, ds_info)
```
`tfds.show_examples()` 를 통해서 각 이미지의 클래스와 그 인덱스 (index) 를 확인할 수 있다.
<br></br>
> 학습 데이터셋 외에 평가용 데이터셋도 확인
```python
tfds.show_examples(ds_test, ds_info)
```
<br></br>

## CAM, Grad - CAM 용 모델 준비하기 (02). 물체의 위치정보

> 데이터셋의 메타 정보를 조회해 원본 이미지의 물체 위치정보 확인
```python
ds_info.features
```
`df_info` 를 조회해 `features` 가 어떻게 구성되어 있는지 확인할 수 있다.

`image` 와 `label` 은 입력 이미지와 이미지에 해당하는 정답 클래스의 인덱스이며, `bbox` (바운딩 박스) 는 아래 그림과 같이 물체의 위치를 사각형 영역으로 표기하는 방법이다. `bbox` 는 `BBoxFeature` 이라는 타입으로 정의되어 있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-3-P-2.jpg)
<br></br>

바운딩 박스를 표시하는 방법은 매우 다양하다. 일반적으로 사용되는 방법은 'xywh' 또는 'minmax'로 표기하는 방법이다.

+ '**xywh**' 는 바운딩박스 중심점을 x, y 로 표기하고, 사각형의 너비 w 와 높이 h 를 표기하는 방법이다.
	-   (예)  `(x_center, y_center, width, height)`
	-   x, y 가 중심점이 아니라 좌측 상단의 점을 가리킬 수도 있다.

+ '**minmax**' 는 바운딩박스를 이루는 좌표의 최소값과 최대값을 통해 표기하는 방법이다.
	-   (예)  `(x_min, x_max, y_min, y_max)`
	-   좌표의 절대값이 아니라, 전체 이미지의 너비와 높이를 기준으로 normalize 한 상대적인 값을 표기하는 것이 일반적이다.

-  위 두가지 뿐만 아니라 이미지의 상하좌우 끝단으로부터 거리로 표현하는 방법, 좌우측의 x 값과 상하측의 y값 네 개로 표시하는 방법 (LRTB), 네 점의 x, y 좌표 값을 모두 표시하는 방법 (QUAD) 등 여러 가지 방법이 있다. 따라서 새로운 데이터셋을 접하거나 라이브러리를 활용하실 때는 간단한 바운딩 박스 정보라도 한 번 더 표기법을 확인하는 것이 좋다.

데이터셋에서 `BBoxFeature` 타입의 `bbox` 는 minmax 를 통해 표현하고 있다.  `tfds` 의 경우 height 를 첫번째 축으로로 삼고있어 [minY, minX, maxY, maxX] 를 의미한다. 
<br></br>

## CAM, Grad - CAM 용 모델 준비하기 (03). CAM 을 위한 모델 만들기

이미지넷 데이터로 Pretrained 된 ResNet - 50 모델을 기반으로 이후 Pooling Layer 뒤에 소프트맥스 레이어를 붙여 Grad - CAM 모델을 구현해보자. 이때 소프트맥스 레이어는 소프트맥스 함수를 활성화 함수로 사용하는 Fully Connected Later 이다.
<br></br>
> Pretrained 된 `resnet50` 모델의 활용해 CAM 을 구현하기 위한 기본 모델 구성
```python
num_classes = ds_info.features["label"].num_classes
base_model = keras.applications.resnet.ResNet50(
    include_top=False,     # Imagenet 분류기  fully connected layer 제거
    weights='imagenet',
    input_shape=(224, 224,3),
    pooling='avg',      # GAP를 적용  
)
x = base_model.output
preds = keras.layers.Dense(num_classes, activation = 'softmax')(x)
cam_model=keras.Model(inputs=base_model.input, outputs=preds)
```
CAM 모델은 분류 문제를 위한 모델과 마지막에 Fully Connected Layer 대신 GAP 을 사용하는 것 외에는 동일하다.

`keras.application` 의 `ResNet50` 의 매개변수 `pooling` 에 `'avg'` 를 매개변수로 전달함으로써 쉽게 GAP 연산을 붙일 수 있다.

+ 참고 : [tf.keras.applications.ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
<br></br>
> 구현한 CAM 모델 확인
```python
cam_model.summary()
```
CAM 은 특성맵을 사용해 만든다. 위에서 구현한 CAM 은 `conv5_block3_out` 의 output 이 특성맵이 된다.

이 특성맵에 GAP 를 붙여 특성의 크기를줄이고, 줄어든 특성 전체에 Dense Layer 를 붙여 분류를 수행한다.
<br></br>

## CAM, Grad - CAM 용 모델 준비하기 (04). CAM 모델 학습하기

> 학습 데이터와 검증 데이터에 normalizing 과 resizing을 포함한 간단한 전처리 수행하는 함수 생성
```python
def normalize_and_resize_img(input):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(input['image'], [224, 224])
    input['image'] = tf.cast(image, tf.float32) / 255.
    return input['image'], input['label']

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(
        normalize_and_resize_img, 
        num_parallel_calls=2
    )
    ds = ds.batch(batch_size)
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
```
학습 데이터와 검증 데이터에 normalizing 과 resizing을 포함한 간단한 전처리를 `normalize_and_resize_img()` 에서 수행하며, 이를 포함하여 `apply_normalize_on_dataset()`에서 배치를 구성한다.

`input` 에 이전과 다르게 `bbox` 정보가 포함되어있지만, 지금 수행해야 할 CAM 모델의 학습에는 필요가 없으므로 `normalize_and_resize_img` 과정에서 제외해 주었다. 

CAM 모델은 Object Detection 이나 Segmentation 에도 활용될 수 있지만 Bounding Box 같은 직접적인 라벨을 사용하지 않고 Weakly Supervised Learning 을 통해 물체 영역을 간접적으로 학습시기는 방식이기 때문이다.
<br></br>
> 데이터셋 전처리 및 배치처리 적용
```python
# 데이터셋에 전처리와 배치처리를 적용합니다. 
ds_train_norm = apply_normalize_on_dataset(ds_train)
ds_test_norm = apply_normalize_on_dataset(ds_test)

# 구성된 배치의 모양을 확인해 봅니다. 
for input in ds_train_norm.take(1):
    image, label = input
    print(image.shape)
    print(label.shape)
```
<br></br>
> 모델 정의
```python
tf.random.set_seed(2021)
cam_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01),
    metrics=['accuracy'],
)
```
<br></br>
> 모델학습
```python
history_cam_model = cam_model.fit(
    ds_train_norm,
    steps_per_epoch=int(ds_info.splits['train'].num_examples/16),
    validation_steps=int(ds_info.splits['test'].num_examples/16),
    epochs=15,
    validation_data=ds_test_norm,
    verbose=1,
    use_multiprocessing=True,
)
```
<br></br>
> 학습시킨 가중치 저장
```python
import os

cam_model_path = os.getenv('HOME')+'/aiffel/class_activation_map/cam_model.h5'
cam_model.save(cam_model_path)
print("저장 완료!")
```
<br></br>

## CAM 

> 앞서 구현한 모델에 필요 코드들을 한번에 정리
```python
# 커널 재시작 이후 실습을 위해, 이전 스텝의 코드를 모아서 한꺼번에 실행합니다.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import copy
import cv2
from PIL import Image
import urllib3
urllib3.disable_warnings()
(ds_train, ds_test), ds_info = tfds.load(
    'cars196',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)

def normalize_and_resize_img(input):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(input['image'], [224, 224])
    input['image'] = tf.cast(image, tf.float32) / 255.
    return input['image'], input['label']

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(
        normalize_and_resize_img, 
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
> 학습된 모델에서 CAM 생성
```python
def get_one(ds):
    ds = ds.take(1)
    sample_data = list(ds.as_numpy_iterator())
    bbox = sample_data[0]['bbox']
    image = sample_data[0]['image']
    label = sample_data[0]['label']
    return sample_data[0]
```
CAM 생성 작업은 데이터셋 배치 단위가 아니라 개별 이미지 데이터 단위로 이루어지기 때문에, `get_one()`  함수로 데이터셋에서 한 장씩 뽑을 수 있도록 해주었다.
<br></br>
> 데이터 준비
```python
item = get_one(ds_test)
print(item['label'])
plt.imshow(item['image'])
```
<br></br>
> CAM 을 생성하기 위해 이전 스텝에서 학습한 모델 불러오기
```python
import os
cam_model_path = os.getenv('HOME')+'/aiffel/class_activation_map/cam_model.h5'
cam_model = tf.keras.models.load_model(cam_model_path)
```
CAM 을 생성하기 위해서는 (1) 특성 맵, (2) 클래스 별 확률을 얻기 위한 소프트맥스 레이어의 가중치, 그리고 (3) 원하는 클래스의 출력값 이 필요하다. 

이미지에서 모델이 어떤 부분을 보는지 직관적으로 확인하기 위해서 네트워크에서 나온 CAM 을 입력 이미지 사이즈와 같게 만들어 함께 시각화 해야한다. 이를 고려해 `model` 과 `item`을 받았을 때 입력 이미지와 동일한 크기의 CAM 을 반환하는 함수를 만들어야 합니다.
<br></br>
> `model` 과 `item` 을 받았을 때 입력 이미지와 동일한 크기의 CAM을 반환하는 함수 생성
```python
def generate_cam(model, item):
    item = copy.deepcopy(item)
    width = item['image'].shape[1]
    height = item['image'].shape[0]
    
    img_tensor, class_idx = normalize_and_resize_img(item)
    
    # 학습한 모델에서 원하는 Layer의 output을 얻기 위해서 모델의 input과 output을 새롭게 정의해줍니다.
    # model.layers[-3].output에서는 우리가 필요로 하는 GAP 이전 Convolution layer의 output을 얻을 수 있습니다.
    cam_model = tf.keras.models.Model([model.inputs], [model.layers[-3].output, model.output])
    conv_outputs, predictions = cam_model(tf.expand_dims(img_tensor, 0))
    
    conv_outputs = conv_outputs[0, :, :, :]
    class_weights = model.layers[-1].get_weights()[0] #마지막 모델의 weight activation을 가져옵니다.
    
    cam_image = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights[:, class_idx]):
        # W * f 를 통해 class별 activation map을 계산합니다.
        cam_image += w * conv_outputs[:, :, i]

    cam_image /= np.max(cam_image) # activation score를 normalize합니다.
    cam_image = cam_image.numpy()
    cam_image = cv2.resize(cam_image, (width, height)) # 원래 이미지의 크기로 resize합니다.
    return cam_image
```
`generate_cam()` 을 구현하기 위해 아래에서는 `conv_ouputs` 와 같이 특정 레이어의 결과값을 output 으로 받기 위해 새로운 모델을 정의하고, feedforward 를 거친 후 CAM 을 계산하도록 구현하였으며, 마지막에는 입력 이미지의 크기에 맞춰 CAM 을 `resize` 해주었다.
<br></br>
> 위에서 생성한 함수를 실행하여 CAM 이미지 도출
```python
cam_image = generate_cam(cam_model, item)
plt.imshow(cam_image)
```
<br></br>
> CAM 이미지를 원본 이미지를 합치는 함수 생성
```python
def visualize_cam_on_image(src1, src2, alpha=0.5):
    beta = (1.0 - alpha)
    merged_image = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
    return merged_image
```
<br></br>
> CAM 이미지와 원본 이미지 합치기
```python
origin_image = item['image'].astype(np.uint8)
cam_image_3channel = np.stack([cam_image*255]*3, axis=-1).astype(np.uint8)

blended_image = visualize_cam_on_image(cam_image_3channel, origin_image)
plt.imshow(blended_image)
```
생성된 CAM 이 차종을 식별하는데 중요한 이미지 부분을 잘 포착하는데 주로 차량의 전면 엠블럼이 있는 부분이 강조되는 경향이 있는데, 이것은 사람이 차종을 식별할 때 유의해서 보는 부분과 일맥상통하다는 것을 확인할 수 있다.
<br></br>

## Grad - CAM

> CAM 모델의 `cam_model`을 그대로 활용해 새로운 이미지 뽑기
```python
item = get_one(ds_test)
print(item['label'])
plt.imshow(item['image'])
```
<br></br>

이제 Grad - CAM 을 구현해보자. 

<br></br>
> Grad - CAM 구현 
```python
def generate_grad_cam(model, activation_layer, item):
    item = copy.deepcopy(item)
    width = item['image'].shape[1]
    height = item['image'].shape[0]
    img_tensor, class_idx = normalize_and_resize_img(item)
    
    # Grad cam에서도 cam과 같이 특정 레이어의 output을 필요로 하므로 모델의 input과 output을 새롭게 정의합니다.
    # 이때 원하는 레이어가 다를 수 있으니 해당 레이어의 이름으로 찾은 후 output으로 추가합니다.
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(activation_layer).output, model.output])
    
    # Gradient를 얻기 위해 tape를 사용합니다.
    with tf.GradientTape() as tape:
        conv_output, pred = grad_model(tf.expand_dims(img_tensor, 0))
    
        loss = pred[:, class_idx] # 원하는 class(여기서는 정답으로 활용) 예측값을 얻습니다.
        output = conv_output[0] # 원하는 layer의 output을 얻습니다.
        grad_val = tape.gradient(loss, conv_output)[0] # 예측값에 따른 Layer의 gradient를 얻습니다.

    weights = np.mean(grad_val, axis=(0, 1)) # gradient의 GAP으로 class별 weight를 구합니다.
    grad_cam_image = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        # 각 class별 weight와 해당 layer의 output을 곱해 class activation map을 얻습니다.
        grad_cam_image += w * output[:, :, k]
        
    grad_cam_image /= np.max(grad_cam_image)
    grad_cam_image = grad_cam_image.numpy()
    grad_cam_image = cv2.resize(grad_cam_image, (width, height))
    return grad_cam_image
```
`grad_cam`은 관찰을 원하는 레이어와 정답 클래스에 대한 예측값 사이의 그래디언트를 구하고, 여기에 GAP 연산을 적용함으로써 관찰 대상이 되는 레이어의 채널별 가중치를 구한다. 최종 CAM 이미지를 구하기 위해서는 레이어의 채널별 가중치 (`weights`) 와 레이어에서 나온 채널별 특성 맵을 가중합 해주어 `cam_image` 를 얻게 된다.

앞서 CAM 함수와 달리 Grad-CAM 은 어떤 레이어든 CAM 이미지를 뽑아낼 수 있으므로, 그래디언트 계산을 원하는 관찰 대상 레이어 `activation_layer` 를 뽑아서 쓸 수 있도록 `activation_layer` 의 이름을 받고 이를 활용한다.

`generate_grad_cam()`에서는 원하는 레이어의 `output`과 특정 클래스의 prediction 사이의 그래디언트 `grad_val`을 얻고 이를 `weights`로 활용한다.
<br></br>
> 위에서 구현한 Grad - CAM 에서 CAM 이미지를 뽑아 확인
```python
grad_cam_image = generate_grad_cam(cam_model, 'conv5_block3_out', item)
plt.imshow(grad_cam_image)
```
레이어의 이름은 `cam_model.summary()` 에서 찾을 수 있다.
<br></br>
```python
grad_cam_image = generate_grad_cam(cam_model, 'conv4_block3_out', item)
plt.imshow(grad_cam_image)
```
<br></br>
```python
grad_cam_image = generate_grad_cam(cam_model, 'conv3_block3_out', item)
plt.imshow(grad_cam_image)
```
<br></br>


## Detection with CAM

### 바운딩 박스
CAM 에서 물체의 위치는 찾는 Detection 을 해보자.
<br></br>
> 
```python
item = get_one(ds_test)
print(item['label'])
plt.imshow(item['image'])
```
<br></br>
> 앞서 구현한 `generate_cam`을 활용해서 CAM 이미지 추출
```python
cam_image = generate_cam(cam_model, item)
plt.imshow(cam_image)
```
<br></br>
> `get_bbox()` 함수는 바운딩 박스를 만드는 함수 생성
```python
def get_bbox(cam_image, score_thresh=0.05):
    low_indicies = cam_image <= score_thresh
    cam_image[low_indicies] = 0
    cam_image = (cam_image*255).astype(np.uint8)
    
    contours,_ = cv2.findContours(cam_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rotated_rect = cv2.minAreaRect(cnt)
    rect = cv2.boxPoints(rotated_rect)
    rect = np.int0(rect)
    return rect
```
`get_bbox()` 함수는 바운딩 박스를 만들기 위해서 `score_thresh` 를 받아 역치값 이하의 바운딩 박스는 없앤다. 그리고 OpenCV 의 `findContours()` 와 `minAreaRect()` 로 사각형을 찾는다.

이때 `rotated_rect` 라는 회전된 바운딩 박스를 얻을 수 있으며, `boxPoints()`로 이를 꼭지점으로 바꿔준다. 마지막에는 `int` 자료형으로 변환해 준다.
<br></br>
> `cam_image`를 통해 `bbox`를 얻고 이를 이미지 위에 시각화한 모습 확인
```python
image = copy.deepcopy(item['image'])
rect = get_bbox(cam_image)
rect
```
<br></br>
```python
image = cv2.drawContours(image,[rect],0,(0,0,255),2)
plt.imshow(image)
```
<br></br>

### Interesction Over Union

CAM 과 Grad - CAM 두 가지 방법을 통해 바운딩 박스를 얻을 수 있었다. 이 바운딩 박스와 정답 데이터를 비교하여 평가하는 방법은 IoU (Intersection over Union) 이 있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-P-3.max-800x600.jpg)
<br></br>
위 그림은 IoU 를 어떻게 구하는지 나타내는 것으로써, IoU 는 두 개 영역의 합집합인 "union" 영역으로 교집합 영역인 "intersection" 영역의 넓이를 나눠준 값이다.

이를 통해 찾고자 하는 물건의 절대적인 면적과 상관없이, 영역을 정확히 잘 찾아냈는지에 대한 상대적인 비율을 구할 수 있으며, 이를 통해 모델이 영역을 잘 찾았는지를 검증할 수 있는 좋은 지표로써 활용된다.

+ 참고 : [C4W3L06 Intersection Over Union](https://www.youtube.com/watch?v=ANIzQ5G-XPE)
<br></br>
> 바운딩 박스의 좌표를 구하는 함수 생성
```python
# rect의 좌표는 (x, y) 형태로, bbox는 (y_min, x_min, y_max, x_max)의 normalized 형태로 주어집니다. 
def rect_to_minmax(rect, image):
    bbox = [
        rect[:,1].min()/float(image.shape[0]),  #bounding box의 y_min
        rect[:,0].min()/float(image.shape[1]),  #bounding box의 x_min
        rect[:,1].max()/float(image.shape[0]), #bounding box의 y_max
        rect[:,0].max()/float(image.shape[1]) #bounding box의 x_max
    ]
    return bbox
```
<br></br>
> `rect` 를 minmax `bbox`  형태로 치환
```python
pred_bbox = rect_to_minmax(rect, item['image'])
pred_bbox
```
<br></br>
> 데이터의 Ground truth bbox 확인
```python
item['bbox']
```
<br></br>
> CAM 을 통해 얻은 bbox 와 Ground truth bbox 의 유사도를 IoU 를 통해 확인하는 함수 생성
```python
def get_iou(boxA, boxB):
    y_min = max(boxA[0], boxB[0])
    x_min= max(boxA[1], boxB[1])
    y_max = min(boxA[2], boxB[2])
    x_max = min(boxA[3], boxB[3])
    
    interArea = max(0, x_max - x_min) * max(0, y_max - y_min)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
```
<br></br>
> CAM 을 통해 얻은 bbox 와 Ground truth bbox 의 유사도를 IoU 를 통해 확인
```python
get_iou(pred_bbox, item['bbox'])
```
<br></br>
