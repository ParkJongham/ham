# 06. GO / STOP! - Object Detection 시스템 만들기


## 실습목표

1.  바운딩 박스(bounding box) 데이터셋을 전처리할 수 있습니다.
2.  Object detection 모델을 학습할 수 있습니다.
3.  Detection 모델을 활용한 시스템을 만들 수 있습니다.


## 학습내용

1.  자율주행 보조장치
2.  RetinaNet
3.  keras-retinanet
4.  프로젝트: 자율주행 보조 시스템 만들기


## 사전 준비

1. Tensorflow 2.3.0 version
	```python
	# 텐서플로우 버전 재설치
	$ pip uninstall tensorflow 
	$ pip install tensorflow==2.3.0
	```
2. tf.keras 2.4.0 version
3. 프로젝트 디렉토리의 repositoty 및 케라스 라이브러리 설치
	```python
	# 작업 디렉토리
	$ cd ~/aiffel/object_detection 
	
	$ git clone https://github.com/fizyr/keras-retinanet.git 
	$ cd keras-retinanet && python setup.py build_ext --inplace

	# keras-retinanet 설치
	$ pip install tensorflow_datasets tqdm 
	$ pip install -r requirements.txt 
	$ pip install .
	```

<br></br>
## How to make?

일반적으로 딥러닝 기술은 

1. __데이터 준비__
2. __딥러닝 네트워크 설계__
3. __학습__ 
4. __테스트 (평가)__

의 과정을 따르게 된다.


## 자율주행 보조장치 (1). KITTI 데이터셋

자율주행 보조장치는 사람 및 차량 등의 물체가 가까이 탐지된다면 멈춰야한다. 

KITTI 데이터셋은 자율주행을 위한 데이터셋으로 2D 와 3D object detection 라벨을 제공한다.

+ [참고 : KITTI 데이터셋](http://www.cvlibs.net/datasets/kitti/)

<br></br>
> 필요라이브러리 가져오기
```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow_datasets as tfds

import copy
import cv2
from PIL import Image, ImageDraw
```
<br></br>
> KITTI 데이터셋 다운로드
```python
# 다운로드에 매우 긴 시간이 소요됩니다. 
import urllib3
urllib3.disable_warnings()
(ds_train, ds_test), ds_info = tfds.load(
    'kitti',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)
```
<br></br>
> KITTI 데이터셋 살펴보기 
> (`tfds.show_examples` 매서드 사용)
```python
fig = tfds.show_examples(ds_train, ds_info)
```
일반적인 사진에 비해 광각으로 촬영되어 물체를 다양한 각도로 확인가능
<br></br>
> KITTI 데이터셋의 정보를 확인
```python
ds_info
```
<br></br>


## 자율주행 보조장치 (2). 데이터 직접 확인하기

> KITTI 데이터셋 확인, 데이터셋의 이미지와 라벨 확인
> (`ds_train.take(1)` 매서드 사용)
```python
TakeDataset = ds_train.take(1)

for example in TakeDataset:  
    print('------Example------')
    print(list(example.keys())) # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
    image = example["image"]
    filename = example["image/file_name"].numpy().decode('utf-8')
    objects = example["objects"]

print('------objects------')
print(objects)

img = Image.fromarray(image.numpy())
img
```
`ds_train.take(1)` 매서드는 데이터셋을 하나씩 뽑아볼 수 있으며, 이미지와 라벨과 같은 정보를 확인할 수 있다.
<br></br>
> 확인한 이미지의 바운딩 박스 (bounding box) 확인
> ([참고 : Pillow 라이브러리의 `ImageDraw` 모듈](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html) 사용)
```python
# 이미지 위에 바운딩 박스를 그려 화면에 표시해 주세요.
def visualize_bbox(input_image, object_bbox):
    input_image = copy.deepcopy(input_image)
    draw = ImageDraw.Draw(input_image)

    # 바운딩 박스 좌표(x_min, x_max, y_min, y_max) 구하기
    width, height = img.size
    print('width:', width, ' height:', height)
    print(object_bbox.shape)
    x_min = object_bbox[:,1] * width
    x_max = object_bbox[:,3] * width
    y_min = height - object_bbox[:,0] * height
    y_max = height - object_bbox[:,2] * height

    # 바운딩 박스 그리기
    rects = np.stack([x_min, y_min, x_max, y_max], axis=1)
    for _rect in rects:
        print(_rect)
        draw.rectangle(_rect, outline=(255,0,0), width=2)
    print(input_image)
    return input_image

visualize_bbox(img, objects['bbox'].numpy())
```
KITTI 에서 제공하는 데이터셋 설명은 다음과 같다.
![](https://github.com/ParkJongham/ham/blob/master/AIFFEL/GOING%20DEEPER/GOING%20DEEPER_06/img/KITTI%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%20%EC%84%A4%EB%AA%85.png?raw=true)
<br></br>


## RetinaNet

RetinaNet 은 Focal Loss for Dense Object Detection 논문을 통해 공개된 detection 모델이다.

+ 참고 : 
	+ [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
	+ [kimcando94님의 Object Detection에 대하여_01: Overall Object detection flow](https://kimcando94.tistory.com/115)
	+ [김홍배님의 Focal loss의 응용(Detection & Classification)](https://www.slideshare.net/ssuser06e0c5/focal-loss-detection-classification)

여기서는 미리 구현된 모델 라이브러리를 가져와 커스텀 데이터셋에 학습하여 사용한다.

1 - stage detector 모델인 YOLO 와 SSD 는 2 - stage detector 모델들 (Faster - RCNN 등) 속도는 빠르지만 성능이 비교적 낮았다.

RetinaNet 은 이 성능 문제를 개선하기 위해 focal loss, FPN (Feature Pyramid Network) 를 적용하여 개선했다.

 <br></br>
### Focal Loss

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-P-car.max-800x600.png)

기존의 1 - stage detection 모델들이 모든 그리드 (grid) 에 대해 한 번에 학습하기때문에 필연적으로 클래스 간 불균형이 발생한다.

Focal loss 는 이런 클래스간 불균형을 해결하기위한 방안으로 사용되었다.

먼저, 위 그림에서 왼쪽 7 x 7 feature level 에서는 한 픽셀이며, 오른쪽의 image level (자동차 사진)에서 보이는 그리드는 각 픽셀의 receptive field 이다.

일반적으로 이미지에서 물체보다는 배경을 더 많이 학습하게 된다.

논문에서는 Focal loss 를 이용해 Loss 를 개선하여 이 문제를 해결하고자 하였다. 즉, 배경의 class 가 많은 class imbalanced data 이며, 배경 class 에 압도되지 않도록 modulating factor 로 손실을 조절한다.

Focal loss 는 교차 엔트로피를 기반으로하며, 교차 엔트로피 $CE (p_t)$ 앞에 $(1 - p_t)^\gamma$  라는 modulating factor 을 붙여준다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/focal_loss.png)

교차엔트로피는 ground truth class 에 대한 확률이 높을 수록 잘 분류된 것으로 판단하며, 때문에 손실이 줄어든다.

하지만 확률이 1 에 매우 가깝지 않는 이상 상당히 큰 손실로 이어지며 이는 object detection 모델 학습에 문제가 될 수 있다.

Focal loss 에서 감마를 0 으로 설정하면 modulation factor 가 0 이 되어 일반적인 교차 엔트로피가 되며, 감마가 커질수록 modulation 이 강하게 적용된다.

<br></br>
### FPN (Feature Pyramid Network)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-4-P-fpn.max-800x600.png)

FPN 은 이름에서 볼 수 있듯이 특성을 피라미트처럼 쌓아서 사용하는 방식이다.

CNN 백본 네트워크에서는 다양한 레이어의 결과값을 특성맵으로 사용할 수 있으며,  컨볼루션 연산은 커널을 통해 일정한 영역을 보고 몇 개의 숫자로 요약한다.

즉, 모델 앞에서 뒷 부분으로 갈 수록 특성맵은 하나의 셀 (cell) 은 보다 넓은 이미지 영역의 정보를 함축적으로 가진다. 이를 receptive field 라고하며, 레이어가 깊어질 수록 pooling 을 거쳐 넓은 범위의 정보를 갖게된다.

FPN 은 백본의 여러 레이어를 한꺼번에 쓴다는 의미를 가지며, SSD 에서 각 레이어의 특석맵에서 다양한 크기에 대한 결과를 얻는 방식이라면 RetinaNet 은 모델 뒷 쪽의 receptive field (특성맵) 을 upsampling 하여 모델 앞단의 특성맵과 더해서 사용한다.

즉, 레이어가 깊어질수록 특성맵의 $w, h$ 방향의 receptive field 가 넓어지는데, 넓게 보는 것과 좁게 보는 것을 같이 쓰겠다는 목적이다.

+ 참고 : [Upsampling](https://www.youtube.com/watch?v=nDPWywWRIRo)

<br></br>
### RetinaNet 논문에서 FPN 구조 적용

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-3-P-FPN.max-800x600.png)

FPN 은 각 level 이 256 개의 채널로 이루어지며, RetinaNet 에서 FPN 은 $P_3 \sim P_7$ 까지의 Pyramid level 을 사용한다.

이를 통해 **Classification Subnet**과 **Box Regression Subnet** 2개의 Subnet을 구성하며, Anchor 갯수를 $A$ 라고 하면 최종적으로 Classification Subnet 은 $K$개 class 에 대해 $KA$ 개 채널을, Box Regression Subnet 은 $4A$ 개 채널을 사용한다.

<br></br>
## Keras-RetinaNet 실습 (1). 데이터 포맷 변경

케라스에서는 RetinaNet 을 이미 구현해 라이브러리로써 사용할 수 있도록 제공한다.

텐서플로우 2버전을 지원하는 repository 가 만들어졌지만 아직 커스텀 데이터셋을 학습하는 방법을 제공하고 있지않다.

따라서 커스텀 데이터셋, KITTI 데이터셋을 사용하기 위해서는 라이브러리 수정이 필요하다.

하지만 해당 모델을 훈련할 수 있는 공통된 데이터셋 포맷을 CSV 형태로 모델 변경해주는 방법이 보다 쉬운 방법이다.

+ 참고 : [Keras RetinaNet](https://github.com/fizyr/keras-retinanet)

<br></br>
### 클래스 및 바운딩 박스 정보 추출

위에서 `tensorflow_dataset` 의 API 를 사용해 이미지와 각 이미지 바운딩 박스의 라벨 정보를 얻었다.

이제 API 를 활용하여 데이터 추출 및 CSV 형태로 한 줄씩 저장해줘야한다.

+ 유의사항 : 
	* 한 라인에 이미지 파일의 위치, 바운딩 박스 위치, 그리고 클래스 정보를 가지는 CSV 파일을 작성하도록 코드를 작성하고, 이를 사용해 CSV 파일을 생성해야한다.
	
	* 브레이크 시스템은 차와 사람을 구분해야 하는 점을 유의

![](https://github.com/ParkJongham/ham/blob/master/AIFFEL/GOING%20DEEPER/GOING%20DEEPER_06/img/%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%20%ED%8F%AC%EB%A7%B7%20%ED%98%95%EC%8B%9D.png?raw=true)

![](https://github.com/ParkJongham/ham/blob/master/AIFFEL/GOING%20DEEPER/GOING%20DEEPER_06/img/csv%20%ED%8F%AC%EB%A7%B7%EC%8B%9C%20%EC%B0%B8%EA%B3%A0%EC%82%AC%ED%95%AD.png?raw=true)

<br></br>
> 클래스 및 바운딩 박스 정보 추출
> (tqdm 은 루프문에 상태에 따라 콘솔에 진행 상황 바를 표시하는 라이브러리이다.)
> 참고 : [tqbm](https://github.com/tqdm/tqdm)
```python
import os
data_dir = os.getenv('HOME')+'/aiffel/object_detection/data'
img_dir = os.getenv('HOME')+'/kitti_images'
train_csv_path = data_dir + '/kitti_train.csv'

# parse_dataset 함수를 구현해 주세요.
def parse_dataset(dataset, img_dir="kitti_images", total=0):
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    # Dataset의 claas를 확인하여 class에 따른 index를 확인해둡니다.
    # 저는 기존의 class를 차와 사람으로 나누었습니다.
    type_class_map = {
        0: "car",
        1: "car",
        2: "car",
        3: "person",
        4: "person",
        5: "person",
    }
    # Keras retinanet을 학습하기 위한 dataset을 csv로 parsing하기 위해서 필요한 column을 가진 pandas.DataFrame을 생성합니다.
    df = pd.DataFrame(columns=["img_path", "x1", "y1", "x2", "y2", "class_name"])
    for item in tqdm(dataset, total=total):
        filename = item['image/file_name'].numpy().decode('utf-8')
        img_path = os.path.join(img_dir, filename)

        img = Image.fromarray(item['image'].numpy())
        img.save(img_path)
        object_bbox = item['objects']['bbox']
        object_type = item['objects']['type'].numpy()
        width, height = img.size

        # tf.dataset의 bbox좌표가 0과 1사이로 normalize된 좌표이므로 이를 pixel좌표로 변환합니다.
        x_min = object_bbox[:,1] * width
        x_max = object_bbox[:,3] * width
        y_min = height - object_bbox[:,2] * height
        y_max = height - object_bbox[:,0] * height

        # 한 이미지에 있는 여러 Object들을 한 줄씩 pandas.DataFrame에 append합니다.
        rects = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.int)
        for i, _rect in enumerate(rects):
            _type = object_type[i]
            if _type not in type_class_map.keys():
                continue
            df = df.append({
                "img_path": img_path,
                "x1": _rect[0],
                "y1": _rect[1],
                "x2": _rect[2],
                "y2": _rect[3],
                "class_name": type_class_map[_type]
            }, ignore_index=True)
            break
    return df

df_train = parse_dataset(ds_train, img_dir, total=ds_info.splits['train'].num_examples)
df_train.to_csv(train_csv_path, sep=',',index = False, header=False)
```
<br></br>
> 테스트 데이터셋에 대해 클래스 및 바운딩 박스 추출
> (`parse_dataset()` 적용하여 dataframe 생성)
```python
test_csv_path = data_dir + '/kitti_test.csv'

df_test = parse_dataset(ds_test, img_dir, total=ds_info.splits['test'].num_examples)
df_test.to_csv(test_csv_path, sep=',',index = False, header=False)
```
<br></br>


### 클래스 맵핑

데이터셋의 클래스는 문자열이지만, 모델은 숫자만을 인식하며, 출력한다. 

학습 후 모델이 숫자로 추론한 클래스를 출력하면 이를 원래 클래스로 바꿔줘야한다. 즉, 어떤 클래스가 있고, 각 클래스는 어떤 인덱스에 맵핑될지 미리 정하고 저장해줘야한다.

> 자동차와 사람을 구별하기 위한 클래스 맵핑 함수 생성
> ![](https://github.com/ParkJongham/ham/blob/master/AIFFEL/GOING%20DEEPER/GOING%20DEEPER_06/img/%EC%9E%90%EB%8F%99%EC%B0%A8%EC%99%80%20%EC%82%AC%EB%9E%8C%EC%9D%84%20%EA%B5%AC%EB%B3%84%ED%95%98%EB%8A%94%20%ED%81%B4%EB%9E%98%EC%8A%A4%20%EB%A7%B5%ED%95%91.png?raw=trueg)
```python
class_txt_path = data_dir + '/classes.txt'

def save_class_format(path="./classes.txt"):
    class_type_map = {
        "car" : 0,
        "person": 1
    }
    with open(path, mode='w', encoding='utf-8') as f:
        for k, v in class_type_map.items():
            f.write(f"{k},{v}\n")

save_class_format(class_txt_path)
```
<br></br>


## Keras-RetinaNet 실습 (2). 셋팅

아래 keras-retinanet 을 참고하여 변환한 데이터셋으로 학습하자.

환경에 따라 `batch_size`, `worker`, `epoch` 를 조절해야한다.

이미지 크기 및 `batch_size` 가 너무 크면 GPU 메모리가 부족할 수 있으므로 조절이 필요하다.

* 원 개발자는 8G 메모리도 RetinaNet을 훈련시키기에는 부족할 수 있다고 한다.

* 참고 : [keras-retinanet](https://github.com/fizyr/keras-retinanet/issues/499)

<br></br>
> RetinaNet 훈련
```python
# RetinaNet 훈련이 시작됩니다!! 50epoch 훈련에 1시간 이상 소요될 수 있습니다. 
!cd ~/aiffel/object_detection && python keras-retinanet/keras_retinanet/bin/train.py --gpu 0 --multiprocessing --workers 4 --batch-size 2 --epochs 50 --steps 195 csv data/kitti_train.csv data/classes.txt
```
<br></br>
> 모델의 추론을 위해 케라스 모델로 변환
```python
!cd ~/aiffel/object_detection && python keras-retinanet/keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_50.h5 snapshots/resnet50_csv_50_infer.h5
```
<br></br>


## Keras - RetinaNet 실습 (3). 시각화

이제 변환한 모델을 load 하고 추론 및 시각화를 해보자.

일정 점수 이하는 제거해줘야하며, 테스트 셋을 다운받아 사용해보자.

* [test_set.zip](https://aiffelstaticprd.blob.core.windows.net/media/documents/test_set.zip)

![](https://github.com/ParkJongham/ham/blob/master/AIFFEL/GOING%20DEEPER/GOING%20DEEPER_06/img/test%20dataset%20%EB%8B%A4%EC%9A%B4.png?raw=true)

<br></br>
> 변환 모델 load
```python
%matplotlib inline

# automatically reload modules when they have changed
%load_ext autoreload
%autoreload 2

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

gpu = '0'
setup_gpu(gpu)

model_path = os.path.join('.', 'snapshots', 'resnet50_csv_50_infer.h5')
model = load_model(model_path, backbone_name='resnet50')
```
<br></br>
> 추론 및 시각화
```python
import os
img_path = os.getenv('HOME')+'/aiffel/object_detection/test_set/go_1.png'

# inference_on_image 함수를 구현해 주세요.
def inference_on_image(model, img_path="./test_set/go_1.png", visualize=True):
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    color_map = {
        0: (0, 0, 255), # blue
        1: (255, 0, 0) # red
    }

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale

    # display images
    if  visualize:
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            print(box)
            if score < 0.5:
                break
            b = box.astype(int)
            draw_box(draw, b, color=color_map[label])

            caption = "{:.3f}".format(score)
            draw_caption(draw, b, caption)

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()            

inference_on_image(model, img_path=img_path)

img_path = os.getenv('HOME')+'/aiffel/object_detection/test_set/stop_1.png'
inference_on_image(model, img_path=img_path)
```
<br></br>
