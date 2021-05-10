# 10. 인물 사진을 만들어 보자.

인물사진 모드란 피사체를 가깝게 찍을 떄 배경이 흐려지는 아웃포커싱 효과를 적용한 것이다.

오늘은 핸드폰 인물 사진 모드를 구현 해보자. 핸드폰 카메라를 통한 인물사진 모드는 서로 다른 화각을 가진 2 개의 렌즈를 통해 일반 (광각) 렌즈에서는 배경을 촬영하고 망원 렌즈에서는 인물을 촬영한 뒤 배경을 흐리게 처리한 후 망원 렌즈의 인물과 적절하게 합성한다.

하지만 딥러닝을 적용해 하나의 렌즈만으로 인물 사진 모드를 만들어보자.

### 인물사진 모드에서 사용되는 용어

한국에서는 배경을 흐리게 하는 기술을 주로 '아웃포커싱'이라고 표현한다. 

하지만 아웃포커싱은 한국에서만 사용하는 용어이고 정확한 영어 표현은  **얕은 피사계 심도 (shallow depth of field)**  또는  **셸로우 포커스 (shallow focus)**  라고 한다.

또한 "보케 (bokeh)"라는 일본어에서 유래된 표현 또한 많이 사용한다.
<br></br>
## 실습 환경 준비

> 실습을 위한 디렉토리 구성
``` bash
$ mkdir -p ~/aiffel/human_segmentation/models 
$ mkdir -p ~/aiffel/human_segmentation/images
```
<br></br>

##  셸로우 포커스 만들기 (01). 사진을 준비하자.

### 하나의 카메라로 셸로우 포커스 (Shallow Focus) 를 만드는 방법

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-14-2.max-800x600_jKvxGUl.png)
<br></br>
두 개의 렌즈가 맡은 역할을 하나의 렌즈에서 구현해야 한다. 이를 위한 딥러닝 기법으로는 이미지 세그멘테이션 (image segmentation) 기술이 있다.

이미지 세그멘테이션은 하나의 이미지에서 배경과 사람을 분리할 수 있는 기술이다. 분리된 배경을 블러 (blur) 처리 한 후 사람 이미지와 다시 하비면 아웃포커싱 효과를 적용한 인물 사진 모드를 구현할 수 있다.

즉, 인물 사진 모드를 위한 과정은 다음과 같다.

1.  배경이 있는 셀카를 촬영한다. (배경과 사람의 거리가 약간 멀리 있으면 좋다.)

2.  시멘틱 세그멘테이션 (Semantic segmentation) 으로 피사체 (사람) 와 배경을 분리한다.

3.  블러링 (blurring) 기술로 배경을 흐리게 한다.

4.  피사체를 배경의 원래 위치에 합성 한다.
<br></br>

### 사진 준비

사진을 준비한 후 실습 디렉토리로 옮기자.

> 사용할 디렉토리의 구조
```bash
- aiffel/human_segmentation 
	├── models 
			└── deeplab_model.tar.gz (미리 준비할 필요는 없습니다.) 
	├── images 
			└── my_image.png (사진 이름은 각자 파일명에 따라 다르게 사용하시면 됩니다)
```
<br></br>
> 사용할 라이브러리 설치 및 가져오기
```python
pip install opencv-python

import cv2
import numpy as np
import os
from glob import glob
from os.path import join
import tarfile
import urllib

from matplotlib import pyplot as plt
import tensorflow as tf
```
`urllib` 패키지는 웹에서 데이터를 다운로드 받을 때 사용한다.
<br></br>
> 준비한 이미지 가져오기
```python
import os
img_path = os.getenv('HOME')+'/human_segmentation/images/my_image.jpg'  # 본인이 선택한 이미지의 경로에 맞게 바꿔 주세요. 
img_orig = cv2.imread(img_path) 
print (img_orig.shape)
```
<br></br>

## 셸로우 포커스 만들기 (02). 세그멘테이션으로 사람 분리하기

### 세그멘테이션 (Segmentation) 이란?

 이미지에서 픽셀 단위로 객체를 추출하는 방법을 의미한다.

이미지 세그멘테이션은 모든 픽셀에 라벨 (label) 을 할당하고 같은 라벨을 공통적인 특징을 가진다고 가정한다.

이 때 공통 특징은 물리적 의미가 없을 수도 있다. 픽셀이 비슷하게 생겼다는 사실은 인식하지만, 실체 물체 단위로 인식하지 않을 수 있는 것이다.

세그멘테이션은 2 가지로 나뉘는데 시멘틱 세그멘테이션 (Semantic Segmentation) 과 인스턴스 세그멘테이션 (Instance Segmentation) 이다.

시멘틱 세그멘테이션이란 물리적 의미 단위로 인식하는 세그멘테이션을 의미한다.

즉, 이미지에서 픽셀을 사람, 자동차, 비행기 등 물리적 단위로 분류하는 방법이다. 하지만 같은 물체가 여러개 있을 경우 각 각을 분리하지 못하며, 하나로 분류한다.

인스턴스 시그멘테이션이란 특정 물체에 관한 추상적인 정보를 이미지에서 추출해내는 방법이다. 따라서 특정 물체에 관계없이 같은 라벨로 표현이 된다.

또한 동일한 특정 물체가 여러개 있을 경우 각 각의 라벨을 가지게 된다. 즉, 물체가 3 명의 사람일 경우 사람 1, 사람 2, 사람 3 으로 구분한다. 이는 시멘틱 세그멘테이션과 명확히 구분되는 차이점이다.
<br></br>
### 이미지 세그멘테이션의 간단한 알고리즘 : 워터쉐드 세그멘테이션 (Watershed Segmentation) 

이미지에서 영역을 분할하는 가장 간단한 방법은 물체의 '경계' 를 나누는 것이다.

이러한 경계는 이미지는 그레이스케일 (grayscale) 로 변환하면 0 ~ 255 의 값을 가지게 되는데, 이 픽셀 값을 이용해서 각 위치의 높고 낮음을 구분할 수 있다.

이때 낮은 부분부터 서서히 '물'을 채워 나간다고 생각할 때 각 영역에서 점점 물이 차오르다가 넘치는 시점이 생기게된다. 그 부분을 경계선으로 만들면 물체를 서로 구분할 수 있게 된다.

+ 참고 : [opencv-python tutorial](https://opencv-python.readthedocs.io/en/latest/doc/27.imageWaterShed/imageWaterShed.html)
<br></br>

## 셸로우 포커스 만들기 (03). 시맨틱 세그멘테이션 다뤄보기

세그멘테이션 문제에는 FCN, SegNet, U - Net 등 많은 모델이 사용된다.

이 중 DeepLab 라는 세그멘테이션 모델을 만들고 모델에 이미지를 입력함으로써 사용해보자ㅏ.

DeepLab 알고리즘 (DeepLab v3 +) 은 세그멘테이션 모델 중에서도 성능이 좋아 최근까지 많이 사용되는 모델이다.

DeepLab 의 특징은 atrous convolution 을 사용한 것인데, 이를 통해 기존 convolution 과 동일한 양의 파라미터와 계산량을 유지하면서도, field of view (한 픽셀이 볼 수 있는 영역) 를 크게 가져갈 수 있게 된다. 즉 적은 파라미러토 넓은 영역을 보게하기 위함이다.

+ 참고 : [DeepLab V3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/)
<br></br>
> DeepLab 모델 준비
```python
class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    # __init__()에서 모델 구조를 직접 구현하는 대신, tar file에서 읽어들인 그래프구조 graph_def를 
    # tf.compat.v1.import_graph_def를 통해 불러들여 활용하게 됩니다. 
    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        with self.graph.as_default():
            tf.compat.v1.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    # 이미지를 전처리하여 Tensorflow 입력으로 사용 가능한 shape의 Numpy Array로 변환합니다.
    def preprocess(self, img_orig):
        height, width = img_orig.shape[:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(img_orig, target_size)
        resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img_input = resized_rgb
        return img_input
        
    def run(self, image):
        img_input = self.preprocess(image)

        # Tensorflow V1에서는 model(input) 방식이 아니라 sess.run(feed_dict={input...}) 방식을 활용합니다.
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [img_input]})

        seg_map = batch_seg_map[0]
        return cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR), seg_map
```
`preprocess()` 는 전처리, `run()` 은 실제로 세그멘테이션을 하는 함수이다.

먼저 input tensor를 만들기 위해 `preprocess()` 함수에서 이미지를 전처리한다.

모델이 받는 입력 크기가 정해져 있으므로 이에 따라 적절한 크기로 resize 하고, OpenCV 의 디폴트 BGR 채널 순서를 텐서플로우에 맞는 RGB 로 수정하며, 전처리된 이미지는 `run()` 함수에서 입력값으로 사용한다.

-   참고 : [DeepLab Demo](https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)
<br></br>
`tf.compat.v1` 은 Tensorflow V1으로 작성한 DeepLab 모델구조를 Tensorflow V2 에서 V1 코드와 모델구조를 활용할 수 있도록 제공한다.

그래서 다소 생소할 수 있는 `session`, `graph`, `feed_dict` 등 Tensorflow V2 에서는 `Model`, `Input` 등에 감추어져 있는 구조가 위 코드에 있다.
<br></br>
> 사전에 학습된 가중치 (pretrained weight) 불러오기
```python
# define model and download & load pretrained weight
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'

model_dir = os.getenv('HOME')+'/human_segmentation/models'
tf.io.gfile.makedirs(model_dir)

print ('temp directory:', model_dir)

download_path = os.path.join(model_dir, 'deeplab_model.tar.gz')
if not os.path.exists(download_path):
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
                   download_path)

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')
```
구글이 제공하는 `deeplabv3_mnv2_pascal_train_aug_2018_01_29` weight 을 다운로드 받고 `DeepLabModel` 을 초기화 한다. 

이 모델은 PASCAL VOC 2012 라는 대형 데이터셋으로 학습된 v3 버전이다.

+ 참고 : [다양한 데이터셋과 백본 (backbone) 모델에 대한 pretrained weight](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)
<br></br>
> 준비한 이미지를 네트워크에 입력
```python
img_resized, seg_map = MODEL.run(img_orig)
print (img_orig.shape, img_resized.shape, seg_map.max())
```
`img_orig` 의 크기는 `1280x720` 이고 `img_resized` 의 크기는 `513x288` 이 출력됨을 확인할 수 있다.

입력 이미지 크기가 달라지면 resize 크기도 달라지며, cv2 는 채널을 HWC 순서로 표시한다.

세그멘테이션 맵에서 가장 큰 값 (물체로 인식된 라벨 중 가장 큰 값) 을 뜻하는 `seg_map.max()` 는 `20` 이라는 값이 출력된다.
<br></br>
> DeepLab PASCAL VOC 의 라벨 종류
```python
LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
]
len(LABEL_NAMES)
```
background 를 제외하면 20 개의 클래스가 있다.

따라서 위에서 출력된 20 의 의미는 tv 가 된다.
<br></br>
> 사람, `15` 를 가진 영역을 검출하기 위한 마스크 생성 및 시각화
```python
img_show = img_resized.copy()
seg_map = np.where(seg_map == 15, 15, 0) # 예측 중 사람만 추출
img_mask = seg_map * (255/seg_map.max()) # 255 normalization
img_mask = img_mask.astype(np.uint8)
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.35, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()
```
사람을 뜻하는 `15` 외 예측은 0 으로 만들었다. 따라서 예측된 세그멘테이션 이미지 (map) 는 최대값이 15 가 된다.

일반 이미지는 0 ~ 255 까지의 값을 사용해 픽셀을 표현하므로, 세그멘테이션 맵에 표현된 값을 원본 이미지에 그림 형태로 출력하기 위해 255 로 정규화 하며, `applyColorMap()` 함수로 색을 적용하고 이미지를 화면에 출력한다.
<br></br>

## 셸로우 포커스 만들기 (04). 세그멘테이션 결과를 원래 크기로 복원하기

> 세그멘테이션 결과 (mask) 를 원래 크기로 복원
```python
img_mask_up = cv2.resize(img_mask, img_orig.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
_, img_mask_up = cv2.threshold(img_mask_up, 128, 255, cv2.THRESH_BINARY)

ax = plt.subplot(1,2,1)
plt.imshow(img_mask_up, cmap=plt.cm.binary_r)
ax.set_title('Original Size Mask')

ax = plt.subplot(1,2,2)
plt.imshow(img_mask, cmap=plt.cm.binary_r)
ax.set_title('DeepLab Model Mask')

plt.show()
```
DeepLab 모델을 사용하기 위해 이미지 크기를 작게 resize 해서 입력했기 때문에 출력된 resize 된 입력 크기와 같이 나오게 된다.

원래 크기로 복원을 위해 `cv2.resize()` 함수를 이용한다. 크기를 키울 때 보간(`interpolation`) 을 고려해야 한다. 

`cv2.INTER_NEAREST` 를 이용해서 깔끔하게 처리할 수 있지만 더 정확히 확대하기 위해 `cv2.INTER_LINEAR` 를 사용한다.

보간법이란 이미지의 크기를 변경 (scaling) 하는 과정에서 컴퓨터가 사이사이 픽셀값을 채우는 방법이다.

+ 참고 : [opencv-python 문서](https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html)

결과적으로 `img_mask_up` 은 경계가 블러된 픽셀값 0 ~ 255의 이미지를 얻으며, 확실한 경계를 다시 정하기 위해 중간값인 128 을 기준으로 임계값 (`threshold`) 을 설정한다. 128 이하의 값은 0 으로 128 이상의 값은 255 로 만드는 방법이다.
<br></br>

## 셸로우 포커스 만들기 (05). 배경 흐리게 하기

> 배경을 흐리게하기 위해 세그멘테이션 마스크를 이용해 배경만 추출
```python
img_mask_color = cv2.cvtColor(img_mask_up, cv2.COLOR_GRAY2BGR)
img_bg_mask = cv2.bitwise_not(img_mask_color)
img_bg = cv2.bitwise_and(img_orig, img_bg_mask)
plt.imshow(img_bg)
plt.show()
```
`bitwise_not` 함수를 이용하면 이미지가 반전된다.

즉, 배경은 255, 사람은 0 이 된다. 반전된 세그멘테이션 결과를 이용해서 이미지와 `bitwise_and` 연산을 수행하면 배경만 있는 영상을 얻을 수 있다.

-   참고:  [StackOverflow:  `bitwise_not`,  `bitwise_and`  함수를 사용해 이미지 바꾸기](https://stackoverflow.com/questions/32774956/explain-arguments-meaning-in-res-cv2-bitwise-andimg-img-mask-mask)
<br></br>
> 이미지를 블러처리하기
```python
img_bg_blur = cv2.blur(img_bg, (13,13))
plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
plt.show()
```
<br></br>

## 셸로우 포커스 만들기 (06). 흐린 배경과 원본 영상 합성

> 배경과 사람 영상 합치기
```python
img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()
```
세그멘테이션 마스크가 255 인 부분만 원본 영상을 가지고 오고 반대인 영역은 블러된 미지 값을 사용한다.

-   참고:  [numpy.where() 사용법](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
<br></br>
