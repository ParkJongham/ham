# 흐린 사진을 선명하게

이미지 생성형 기술이 적용된 대표적인 사례는 아마도 저해상도 이미지를 고해상도 이미지로 변환하는 Super Resolution 일 것이다.

GAN 관련 기술은 CNN 보다 정밀한 고해상도 이미지를 생성하는데 효과적이지만, 엄청 오랜 학습 시간이 소요된다.


## 학습 전제

-   신경망의 학습 방법에 대한 전반적인 절차를 알고 있어야 합니다.
-   CNN, GAN에 대한 기본적인 개념을 알고 있어야 합니다.
-   Tensorflow에서 모델을 구성할 때 사용하는 Sequential, Functional API에 대해 알고 있어야 합니다.
-   Tensorflow의 GradientTape API를 이용한 학습 절차를 이해할 수 있어야 합니다.  
      

## 학습 목표

-   Super Resolution과 그 과정 수행에 필요한 기본 개념을 이해합니다.
-   Super Resolution에 사용되는 대표적인 2개의 구조(SRCNN, SRGAN)에 대해 이해하고 활용합니다.


## 데이터셋 다운로드

> tensorflow - datasets 라이브러리를 통해 데이터 다운로드
```python
pip install tensorflow-datasets

import tensorflow_datasets as tfds
tfds.load("div2k/bicubic_x4")
```
<br></br>

## Super Resolution 이란??

### Resolution? Super Resolution?

일반전으로 우리는 고해상도 영상을 선호한다. 하지만 고해상도 영상은 저해상도 영상보다 당연히 더욱 큰 크기를 가지며, 따라서 인터넷 환경이 무엇보다도 중요하다.

하지만 아직도 많은 경우에 고해상도 영상을 재생할 때 주변 환경에 맞춰 어느 정도 적당한 화질로 낮추기도 한다. 

하지만 저해상도 영상을 고해상도로 바꿀 수 있는 기술을 적용한다면 어떻까?

Super Resolution (초해상화) 이란 저해상도 영상을 고해상도 영상으로 변환하는 작업이나 그 과정을 의미한다.

이렇게 Super Resolution 을 이용하면 좋지 않은 인터넷 환경에서도 충분히 고해상도의 영상을 시청하거나 활용할 수 있다.

+ 참고 :
	+ [모니터의 핵심, 디스플레이의 스펙 따라잡기](http://blog.lgdisplay.com/2014/03/%eb%aa%a8%eb%8b%88%ed%84%b0-%ed%95%b5%ec%8b%ac-%eb%94%94%ec%8a%a4%ed%94%8c%eb%a0%88%ec%9d%b4%ec%9d%98-%ec%8a%a4%ed%8e%99-%eb%94%b0%eb%9d%bc%ec%9e%a1%ea%b8%b0-%ed%95%b4%ec%83%81%eb%8f%84/)  
    
	+ [림으로 쉽게 알아보는 HD 해상도의 차이](https://blog.lgdisplay.com/2014/07/%EA%B7%B8%EB%A6%BC%EC%9C%BC%EB%A1%9C-%EC%89%BD%EA%B2%8C-%EC%95%8C%EC%95%84%EB%B3%B4%EB%8A%94-hd-%ED%95%B4%EC%83%81%EB%8F%84%EC%9D%98-%EC%B0%A8%EC%9D%B4/)
<br></br>

## Super Resolution 의 활용사례

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-3.white_tower.max-800x600.png)
<br></br>

실제로 Super Resolution 은 오늘날 다양하게 활용되고 있다. 위 그림은 드라마 하얀거탑을 UHD 화질로 향상하여 재방영한 사례이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-4.cctv.max-800x600.png)
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-5.medical.max-800x600.png)
<br></br>
뿐만 아니라 위 그림처럼 CCTV 의 화질을 개선하여 여러 방면으로 활용할 수 있으며, 의료 영상에 접목하여 구조 정보를 상세히 관찰하여 정량적 이미지 분석 및 진단 등 의사 결정에 도움을 줄 수 있다.

일반적으로 고해상도의 의료 영상을 얻기위해서는 오랜 스캔이 필요하므로 방사선 노출 등 환자에게 부작용을 초래할 수 있다.

하지만 Super Resolution 기술을 활용하면 이러한 단점을 극복할 수 있다.
<br></br>

## Super Resolution 을 어렵게 만드는 요인들

이렇게 여러 방면으로 활용이 가능한 Super Resolution 을 위해서는 몇 가지 어려움을 극복해야한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-6.hard_1.max-800x600.png)
<br></br>
첫 번째, 하나의 저해상도 이미지에 대해 여러개의 고해상도 이미지가 나올 수 있다는 것이다.

위 그림은 1 개의 저해상도 이미지에 대응하는 3 개의 고해상도 이미지를 나타낸 것이다.

3 개의 고새항도 이미지는 눈으로는 그 차이를 알기힘들다.

하지만 그림의 맨 아래 확대한 그림을 보면 픽셀 값이 일정하지 않고 다름을 알 수 있다.

이렇게 하나의 저해상도 이미지를 고해상도 이미지로 만드는데 다양한 경우의 수가 있는 것이 Super Resolution 의 큰 특징이며, `ill-posed (inverse) problem`이라 한다.

일반적으로 Super Resolution 모델을 학습시키기 위한 데이터를 구성하는 과정은, 먼저 고해상도 이미지를 준비하고 특정한 처리 과정을 거쳐 저해상도 이미지를 생성하며, 생성된 저해상도 이미지를 입력으로 원래의 고해상도 이미지를 복원하도록 학습을 진행한다.
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/e-22-7.hard_2.png)<br></br>

두 번째는 Super Resolution 의 복잡도이다.

위 그림은 2 x 2 크기의 이미지를 이용해 3 x 3, 4 x 4, 5 x 5 크기의 이미지로 Super Resolution 하는 과정을 나타낸 것이다.

그림에서 녹색으로 나타난 2 x 2 이미지 픽셀(그림의 왼쪽) 을 입력으로 3 x 3, 4 x 4, 5 x 5 크기의 이미지 (그림의 오른쪽) 를 만들 때 새롭게 생성해야하는 정보는 최소 5 개 픽셀 (그림의 회색에 해당하는 부분) 이며, 최대 21 개의 정보를 생성해야 한다.

원래의 정보 (녹색 부분) 만을 가지고 많은 정보 (회색 부분) 을 만들어내는 과정은 그만큼 고난이도이며, 잘못된 정보를 만들어낼 가능성도 높다.

이러한 문제는 원래의 해상도보다 더 높은 해상도로 Super Resolution 할 수록 심해진다.
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-8.hard_3.max-800x600.png)
<br></br>

위 그림의 첫 번째 및 네 번째 이미지는 저해상도 및 고해상도 이미지를 나타내며, 두 번째 및 세 번째 이미지는 각각 다른 딥러닝 모델을 이용해 Super Resolution 하여 생성한 결과 이미지이다.

Super Resolution 을 적용한 결과를 비교하면 둘 중 어떤 결과가 더 고해상도에 가까울까? 

대부분 결과 2 의 이미지가 더 고해상도로 생각할 수 있다.

하지만 결과 1 의 이미지가 더 높은 해상도를 가진다. 이는 그림의 윗 부분 괄호안의 숫자를 통해 확인할 수 있는데, 이 숫자는 Super Resolution 에서 사용되는 두 개의 정량적 평가 결과이다. 이 평가가 높을 수록 고해상도를 의미하는데, 육안으로 봤을 때와 실제 결과가 다를 수 있다는 것이다.

이렇게 평가 지표와 사람의 시각적 관찰에 의한 판단이 일치하지 않을 수 있다는 것이 세 번째 어려운 점이다.
<br></br>

## 가장 쉬운 Super Resolution

Super Resolution 을 수행하기에 가장 쉬운 방식은 Interpolation 을 이용하는 것이다.

Interpolation 이란 알려진 값을 가진 두 점 사이의 어느 지점의 값이 얼마인지를 추정하는 기법이며, 이는 Linear interpolation, Bilinear interpolation, 2 가지 방법으로 나눠진다.

Linear interpolation (선형 보간법) 은 2 개의 값을 이용해 새로운 픽셀을 예측하는 것이다.

Bilinear interpolation (이진 보간법) 선형 보간법을 2 차원으로 확장시켜 4 개의 값을 이용해 새로운 픽셀을 예측하는 것이다.

+ 참고 :
	+  [선형보간법과 삼차보간법, 제대로 이해하자](https://bskyvision.com/789)
	+ [Bilinear interpolation 예제](http://blog.naver.com/dic1224/220882679460)
<br></br>

> OpenCV 라이브러리를 통해 interpolation 적용을 위한 scikit - image 제공하는 예제 이미지 활용
```python
# 라이브러리 설치

pip install opencv-python 
pip install scikit-image

from skimage import data
import matplotlib.pyplot as plt

hr_image = data.chelsea() # skimage에서 제공하는 예제 이미지를 불러옵니다.
hr_shape = hr_image.shape[:2]

print(hr_image.shape) # 이미지의 크기를 출력합니다.

plt.figure(figsize=(6,3))
plt.imshow(hr_image)
```
예시를 위해 scikit - image 의 고양이 이미지를 활용하였으며, 이 이미지의 크기는 (세로 픽셀 수 x 가로 픽셀 수 x 채널 수) 가 [300 x 451 x 3] 이다.
<br></br>
> Opencv 라이브러리의 resize() 를 활용해 이미지 사이즈 조절
```python
import cv2
lr_image = cv2.resize(hr_image, dsize=(150,100)) # (가로 픽셀 수, 세로 픽셀 수)

print(lr_image.shape)

plt.figure(figsize=(3,1))
plt.imshow(lr_image)
```
사진의 해상도가 높기 때문에 저해상도로 낮추기위해 크기를 resize 한다.

'100 x 150 x 3' 크기로 줄입니다. `dsize` 의 설정값에 따라 크기를 조절하는데, 주의할 점은 변환하고자 하는 이미지의 크기를 _(가로 픽셀 수, 세로 픽셀 수)_ 로 지정해줘야 한다.
<br></br>
> resize 한 이미지에 interpolation 적용한 Super Resolution 
```python
bilinear_image = cv2.resize(
    lr_image, 
    dsize=(451, 300), # (가로 픽셀 수, 세로 픽셀 수) 
    interpolation=cv2.INTER_LINEAR # bilinear interpolation 적용
)

bicubic_image = cv2.resize(
    lr_image, 
    dsize=(451, 300), # (가로 픽셀 수, 세로 픽셀 수)
    interpolation=cv2.INTER_CUBIC # bicubic interpolation 적용
)

images = [bilinear_image, bicubic_image, hr_image]
titles = ["Bilinear", "Bicubic", "HR"]

plt.figure(figsize=(16,3))
for i, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(1,3,i+1)
    plt.imshow(image)
    plt.title(title, fontsize=20)
```
`resize()`내의 `interpolation` 설정에 따라 적용 방법을 조절할 수 있으며, Bilinear 및 Bicubic interpolation 을 적용해 '400 x 600 x 3' 크기의 이미지로 크게 변환한다.

+ 참고 : [OpenCV Documentation](https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)
<br></br>
> Super Resolution 을 적용한 이미지와 원래의 고해상도 이미지 시각화
```python
# 특정 영역을 잘라낼 함수를 정의합니다.
def crop(image, left_top, x=50, y=100):
    return image[left_top[0]:(left_top[0]+x), left_top[1]:(left_top[1]+y), :]


# 잘라낼 영역의 좌표를 정의합니다.
left_tops = [(220,200)] *3 + [(90,120)] *3 + [(30,200)] *3

plt.figure(figsize=(16,10))
for i, (image, left_top, title) in enumerate(zip(images*3, left_tops, titles*3)):
    plt.subplot(3,3,i+1)
    plt.imshow(crop(image, left_top))
    plt.title(title, fontsize=20)
```
이미지가 작아 해상도에 큰 차이가 없어 보인다. 

때문에 특정 부위를 잘라내어 시각화하여 차이를 살펴보았는데, 이미지만 크게 만들어줄 뿐 세세한 정보는 거의 찾아볼 수 없다.
<br></br>

## Deep Learning 을 이용한 Super Resolution (01). SRCNN

SRCNN 은 Super Resolution Convoltional Neural Networks 의 약자로 매우 간단한 모델 구조를 사용하면서 기존에 비해 큰 성능 향상을 이뤄낸 모델이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-9.srcnn.max-800x600.png)
<br></br>
위 그림은 SRCNN 의 구조이다. 먼저 저해상도의 이미지를 bicubic interpolation 하여 원하는 크기로 이미지를 늘린다.

SRCNN 은 이렇게 늘린 이미지 (그림의 ILR) 를 입력으로 사용하며, 3 개의 convolutional layer 를 거쳐 고해상도 이미지를 생성한다.

생성된 고해상도 이미지와 실제 고해상도 이미지 사이의 차이를 역전파를 통해 신경망의 가중치를 학습한다.

즉, 3 가지 연산으로 구성되며 손실함수로는 MSE (Mean Squared Error) 을 사용한다. 3 가지 연산은 다음과 같다.

1. Patch extraction and representation : 저해상도 이미지에서 patch들을 추출하는 과정

2. Non-linear mapping : 추출된 다차원의 patch 들을 non-linear 하게 다른 다차원의 patch 들로 매핑하는 과정

3. Reconstruction : 다차원의 patch 들로부터 고해상도 이미지를 복원하는 과정
<br></br>

+ 참고 : [논문리뷰 - SRCNN](https://d-tail.tistory.com/6)
<br></br>

## Deep Learning 을 이용한 Super Resolution (02). SRCNN 이후 제안된 구조들

SRCNN 이후 Super Resolution 에 관해 엄청난 발전을 이뤘다.

### VDSR (Very Deep Super Resolution)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-10.vdsr.max-800x600.png)
<br></br>

VDSR 모델은 SRCNN 과 동일하게 interpolation 을 통해 저해상도 이미지의 크기를 늘려서 입력으로 사용한다. SRCNN 에 비해 증가된 20 개의 convolutional layer 를 사용했고, 마지막 고해상도 이미지 생성 직전에 처음 입력 이미지를 더해주는 residual learning 을 적용하였다.

구조가 깊어진만큼 SRCNN 에 비해 큰 성능 향상을 이룬 모델이다.
<br></br>

### RDN (Residual Dense Network)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-11.rdn.max-800x600.png)
<br></br>
 
 RDN 은 저해상도 이미지가 입력되면 여러 단계의 convolutional layer 를 거치며, 각 레이어에서 나오는 출력을 최대한 활용한다.

위 그림 중 아래 그림과 같이 각각의 convolution layer 출력 결과로 생성된 특징이 화살표를 따라 뒤의 연산에서 재활용된다.
<br></br>

### RCAN (Residual Channel Attention Networks)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-12.rcan.max-800x600.png)
<br></br>

위 그림은 RCAN 의 구조를 나타낸 것으로 더욱 더 복잡해진 구조를 가지고 있다.

앞의 여러 Super Resolution 모델들과의 차별된 점은 convolutional layer 의 결과로 나온 각각의 특징 맵을 대상으로 채널 간 모든 정보가 균일한 중요도를 갖지않고, 일부 중요한 채널에서만 선택적으로 집중하도록 유도한 것이다. (맨 위의 Channel attention 이라 적힌 부분)
<br></br>

## SRCNN 을 이용해 Super Resolution 도전하기

### 데이터 준비하기

> 데이터 준비 및 임의의 한 쌍의 이미지를 시각화
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# 데이터를 불러옵니다.
train, valid = tfds.load(
    "div2k/bicubic_x4", 
    split=["train","validation"],
    as_supervised=True
)

# 시각화를 위해 한 개의 데이터만 선택합니다.
for i, (lr, hr) in enumerate(valid):
    if i == 6: break
    
# 저해상도 이미지를 고해상도 이미지 크기로 bicubic interpolation 합니다.  
hr, lr = np.array(hr), np.array(lr)
bicubic_hr = cv2.resize(
    lr, 
    dsize=(hr.shape[1], hr.shape[0]), # 고해상도 이미지 크기로 설정
    interpolation=cv2.INTER_CUBIC # bicubic 설정
)

# 저해상도 및 고해상도 이미지를 시각화합니다.
plt.figure(figsize=(20,10))
plt.subplot(1,2,1); plt.imshow(bicubic_hr)
plt.subplot(1,2,2); plt.imshow(hr)
```
tensorflow-datasets 라이브러리의 `DIV2K` 데이터셋을 활용하였으며, 해당 데이터셋은 800 개의 학습용 데이터와 100 개의 검증용 데이터셋으로 구성되어 있으며, 저해상도 이미지와 원본 고해상도 이미지가 한 쌍으로 구성되어 있다.

저해상도 이미지를 bicubic interpolation 하여 고해상도 이미지와 동일한 크기로 만들었다.

+ 참고 : [DIV2K datasets](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
<br></br>
> 이미지의 특정 부분을 확대하여 시각화
```python
# 이미지의 특정 부분을 잘라내는 함수를 정의합니다.
def crop(image, left_top, x=200, y=200):
    return image[left_top[0]:(left_top[0]+x), left_top[1]:(left_top[1]+y), :]

# interpolation된 이미지와 고해상도 이미지의 동일한 부분을 각각 잘라냅니다.
left_top = (400, 500)
crop_bicubic_hr = crop(bicubic_hr, left_top)
crop_hr = crop(hr, left_top)

# 잘라낸 부분을 시각화 합니다.
plt.figure(figsize=(15,25))
plt.subplot(1,2,1); plt.imshow(crop_bicubic_hr); plt.title("Bicubic", fontsize=30)
plt.subplot(1,2,2); plt.imshow(crop_hr); plt.title("HR", fontsize=30)
```
이미지의 크기로 인해 선명함의 차이를 알 수 없기 때문에 특정 부분을 잘라 확대하여 확인해 보았다.

Bicubic interpolation 방법을 이용한 결과는 HR 이라 쓰여진 실제 고해상도 이미지와 비교하면 매우 선명하지 않은 것을 확인 할 수 있다.
<br></br>
> 학습에 활용하기 위해 크기가 큰 이미지 일부 영역을 임의로 자르기 (전처리)
```python
import tensorflow as tf

def preprocessing(lr, hr):
    # 이미지의 크기가 크므로 (96,96,3) 크기로 임의 영역을 잘라내어 사용합니다.
    hr = tf.image.random_crop(hr, size=[96, 96, 3])
    hr = tf.cast(hr, tf.float32) / 255.
    
    # 잘라낸 고해상도 이미지의 가로, 세로 픽셀 수를 1/4배로 줄였다가
    # interpolation을 이용해 다시 원래 크기로 되돌립니다.
    # 이렇게 만든 저해상도 이미지를 입력으로 사용합니다.
    lr = tf.image.resize(hr, [96//4, 96//4], "bicubic")
    lr = tf.image.resize(lr, [96, 96], "bicubic")
    return lr, hr

train = train.map(preprocessing).shuffle(buffer_size=10).batch(16)
valid = valid.map(preprocessing).batch(16)
```
SRCNN 은 고해상도 이미지 크기에 맞에 interpolation 이 적용된 이미지를 입력으로 사용한다.

따라서 학습에 활용하기 위해 이미지의 일부 영역을 임의로 잘랐다.
<br></br>
> SRCNN 구현
```python
from tensorflow.keras import layers, Sequential

# 3개의 convolutional layer를 갖는 Sequential 모델을 구성합니다.
srcnn = Sequential()
# 9x9 크기의 필터를 128개 사용합니다.
srcnn.add(layers.Conv2D(128, 9, padding="same", input_shape=(None, None, 3)))
srcnn.add(layers.ReLU())
# 5x5 크기의 필터를 64개 사용합니다.
srcnn.add(layers.Conv2D(64, 5, padding="same"))
srcnn.add(layers.ReLU())
# 5x5 크기의 필터를 64개 사용합니다.
srcnn.add(layers.Conv2D(3, 5, padding="same"))

srcnn.summary()
```
실제 논문과 달리 간단한 과정만으로 SRCNN 을 구현해보았다.
<br></br>
### SRCNN 학습하기

> 최정화 방법 및 손실함수 설정 후 학습
```python
srcnn.compile(
    optimizer="adam", 
    loss="mse"
)

srcnn.fit(train, validation_data=valid, epochs=1)
```
SRCNN의 학습에는 꽤나 오랜시간이 소요되어기 때문에  구현한 SRCNN 의 실행 여부를 알아보기 위해 1 회만 학습해 보았다.
<br></br>

### SRCNN 테스트하기

> 이미 학습완료된 SRCNN 모델 준비
```bash
$ mkdir -p ~/aiffel/super_resolution 
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/srcnn.h5 
$ mv srcnn.h5 ~/aiffel/super_resolution
```
+ 참고 : [학습된 SRCNN 모델 다운로드](https://aiffelstaticprd.blob.core.windows.net/media/documents/srcnn.h5)
<br></br>
> SRCNN 모델 불러오기
```python
import tensorflow as tf
import os

model_file = os.getenv('HOME')+'/super_resolution/srcnn.h5'
srcnn = tf.keras.models.load_model(model_file)
```
<br></br>
> SRCNN 을 사용하는 함수 정의 및 SRCNN 적용
```python
def apply_srcnn(image):
    sr = srcnn.predict(image[np.newaxis, ...]/255.)
    sr[sr > 1] = 1
    sr[sr < 0] = 0
    sr *= 255.
    return np.array(sr[0].astype(np.uint8))

srcnn_hr = apply_srcnn(bicubic_hr)
```
<br></br>
> 일부 영역을 잘라내 시각적으로 비교
```python
# 자세히 시각화 하기 위해 3개 영역을 잘라냅니다.
# 아래는 잘라낸 부분의 좌상단 좌표 3개 입니다.
left_tops = [(400,500), (300,1200), (0,1000)]

images = []
for left_top in left_tops:
    img1 = crop(bicubic_hr, left_top, 200, 200)
    img2 = crop(srcnn_hr , left_top, 200, 200)
    img3 = crop(hr, left_top, 200, 200)
    images.extend([img1, img2, img3])

labels = ["Bicubic", "SRCNN", "HR"] * 3

plt.figure(figsize=(18,18))
for i in range(9):
    plt.subplot(3,3,i+1) 
    plt.imshow(images[i])
    plt.title(labels[i], fontsize=30)
```
bicubic interpolation을 적용하고 이미지 전체를 시각화했을 때 세부적인 선명함이 눈에 띄지 않았기 때문에, 일부 영역을 잘라내어 시각적으로 비교하였으며, 3 개 이미지 (bicubic interpolation 의 결과, SRCNN 의 결과, 원래 고해상도 이미지)를 나란히 시각화 하였다.

결과적으로 보다 선명해진 것을 알 수 있지만 원래 고해상도 이미지에 비해 크게 향상된 것은 아니다.

이는 데이터셋이 비교적 세밀한 구조의 이미지가 많아 SRCNN 과 같이 간단한 구조로는 더 이상 학습되지 않는 것으로 볼 수 있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-13.srcnn_result.max-800x600.png)
<br></br>
실제 논문에서 위 그림과 같이 비교적 간단한 구조의 이미지에 대해서는 만족할 만한 성능을 보여준다.
<br></br>

## Deep Learning 을 이용한 Super  Resolution (03). SRGAN

SRCAN 은 GAN 모델을 활용해 Super Resolution 한 모델이다.

### 다시 한번, GAN

+ 참고 : 
	+ [참고 자료: GAN - 스스로 학습하는 인공지능](https://www.samsungsds.com/global/ko/support/insights/Generative-adversarial-network-AI.html?moreCnt=19&backTypeId=undefined&category=undefined)
	+ [GAN - GAN의 개념과 이해](https://www.samsungsds.com/global/ko/support/insights/Generative-adversarial-network-AI-2.html)
<br></br>

### SRGAN (Super Resolution + GAN)

SRGAN 은 Super Resolution Using a Generative Adversarial Network 의 약자로 GAN 을 활용한 Super Resolution 모델이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-14.srgan.max-800x600.png)
<br></br>
위 그림은 SRGAN 의 구조를 나타낸다. 그림의 Generator Network 와 Discriminator Network 는 생성자와 판별자로 생성자가 저해상도 이미지를 입력으로 받아 고해상도 이미지를 생성해 내면, 판별자는 생성자가 생성한 고해상도 이미지를 실제 고해상도 이미지와 비교하여 진짜가 무엇인지를 판별한다.

GAN 모델 처럼 생성자는 판별자가 무엇이 진짜인지 모르게 최고의 결과를 내고자 하며, 판별자는 생성자가 생성해내는 결과를 구분하고자 한다. 이렇게 경쟁적인 구조덕에 학습이 진행 될 수록 생성자와 판별자 모두가 발전한다.

최종적으로 학습이 완료되었을 때는 생성자가 생성해낸 고해상도 이미지는 판별자도 구분하기 힘들 정도로 뛰어난 품질을 가진 고해상도 이미지가 된다.

SRGAN 은 다음과 같은 특별한 손실함수를 사용한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-15.srgan_loss.max-800x600.png)
<br></br>
위 식을 보면 content loss 와 adversarial loss 로 구성되어 있으며, adversarial loss 는 일반적인 GAN 의 loss 이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-16.srgan_vgg.max-800x600.png)
<br></br>
위 그림은 content loss 를 나타낸 것이다. content loss 는 생성자를 이용해 얻어낸 고해상도 이미지를 실제 고해상도 이미지와 직접 비교하는 것이 아니다.

content loss 는 각 이미지를 이미지넷으로 사전 학습된 VGG 모델의 입력으로 하여 출력되는 특성 맵에서 차이를 계산한다.

SRGAN 은 SRCNN 이 직접 생성된 고해상도 이미지와 실제 고해상도 이미지 둘을 비교하여 loss 를 계산하는 것이 아니다.

이렇게 VGG 를 이용해 content loss 및 adversarial loss 를 합하여 perceptual loss 라는 최종 손실함수를 계산한다.
<br></br>

## SRGAN 을 이용해 Super Resolution 도전하기


### 데이터 준비하기

> 데이터 준비
```python
train, valid = tfds.load(
    "div2k/bicubic_x4", 
    split=["train","validation"],
    as_supervised=True
)
def preprocessing(lr, hr):
    hr = tf.cast(hr, tf.float32) /255.
        
    # 이미지의 크기가 크므로 (96,96,3) 크기로 임의 영역을 잘라내어 사용합니다.
    hr_patch = tf.image.random_crop(hr, size=[96,96,3])
        
    # 잘라낸 고해상도 이미지의 가로, 세로 픽셀 수를 1/4배로 줄입니다
    # 이렇게 만든 저해상도 이미지를 SRGAN의 입력으로 사용합니다.
    lr_patch = tf.image.resize(hr_patch, [96//4, 96//4], "bicubic")
    return lr_patch, hr_patch

train = train.map(preprocessing).shuffle(buffer_size=10).repeat().batch(8)
valid = valid.map(preprocessing).repeat().batch(8)
```
SRCNN 은 저해상도 이미지에 대해 interpolation 하여 고해상도 이미지 크기로 맞춘 후 입력으로 사용하지만 SRGAN 은 해당 과정을 거치지 않는다.
<br></br>

### SRGAN 구현하기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-17.srgan_gene.max-800x600.png)
<br></br>
위 그림은 SRGAN 의 Generator 의 구조를 나타낸 것이다. 그림의 `k9n64s1`라는 표기는 Convolutional layer 내의 hyperparameter 설정에 대한 정보이며, k 는 kernel size, n 은 사용 필터의 수, s 는 stride 를 나타낸다.

Tensorflow 로 구현한다면 `Conv2D(filters=64, kernel_size=9, strides=1, padding="same")` 처럼 작성할 수 있으며, 모든 stride 가 1 인 convolutional layer 에는 패딩을 통해 출력의 크기를 계속 유지한다.

SRGAN의 Generator에는 skip-connection을 가지고 있으며, 이는 Sequential API로 구현할 수 없기에 직접 구현해야한다.
<br></br>
> SRGAN 의 skip - connection 구현
```python
from tensorflow.keras import Input, Model

# 그림의 파란색 블록을 정의합니다.
def gene_base_block(x):
    out = layers.Conv2D(64, 3, 1, "same")(x)
    out = layers.BatchNormalization()(out)
    out = layers.PReLU(shared_axes=[1,2])(out)
    out = layers.Conv2D(64, 3, 1, "same")(out)
    out = layers.BatchNormalization()(out)
    return layers.Add()([x, out])

# 그림의 뒤쪽 연두색 블록을 정의합니다.
def upsample_block(x):
    out = layers.Conv2D(256, 3, 1, "same")(x)
    # 그림의 PixelShuffler 라고 쓰여진 부분을 아래와 같이 구현합니다.
    out = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(out)
    return layers.PReLU(shared_axes=[1,2])(out)
    
# 전체 Generator를 정의합니다.
def get_generator(input_shape=(None, None, 3)):
    inputs = Input(input_shape)
    
    out = layers.Conv2D(64, 9, 1, "same")(inputs)
    out = residual = layers.PReLU(shared_axes=[1,2])(out)
    
    for _ in range(5):
        out = gene_base_block(out)
    
    out = layers.Conv2D(64, 3, 1, "same")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Add()([residual, out])
    
    for _ in range(2):
        out = upsample_block(out)
        
    out = layers.Conv2D(3, 9, 1, "same", activation="tanh")(out)
    return Model(inputs, out)
```
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-18.srgan_disc.max-800x600.png)
<br></br>
위 그림은 SRGAN 의 Discriminator 의 구조를 나타내며, Generator 의 skip - connection 과 같이 Funcional API 를 통해 직접 구현해야 한다.
<br></br>
> Discriminator 구현
```python
# 그림의 파란색 블록을 정의합니다.
def disc_base_block(x, n_filters=128):
    out = layers.Conv2D(n_filters, 3, 1, "same")(x)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(n_filters, 3, 2, "same")(out)
    out = layers.BatchNormalization()(out)
    return layers.LeakyReLU()(out)

# 전체 Discriminator 정의합니다.
def get_discriminator(input_shape=(None, None, 3)):
    inputs = Input(input_shape)
    
    out = layers.Conv2D(64, 3, 1, "same")(inputs)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(64, 3, 2, "same")(out)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)
    
    for n_filters in [128, 256, 512]:
        out = disc_base_block(out, n_filters)
    
    out = layers.Dense(1024)(out)
    out = layers.LeakyReLU()(out)
    out = layers.Dense(1, activation="sigmoid")(out)
    return Model(inputs, out)
```
<br></br>
> Tensorflow 에서 제공하는 학습된 VGG 19 모델 불러오기
```python
from tensorflow.python.keras import applications
def get_feature_extractor(input_shape=(None, None, 3)):
    vgg = applications.vgg19.VGG19(
        include_top=False, 
        weights="imagenet", 
        input_shape=input_shape
    )
    # 아래 vgg.layers[20]은 vgg 내의 마지막 convolutional layer 입니다.
    return Model(vgg.input, vgg.layers[20].output)
```
SRGAN 은 VGG 19 를 이용해 content loss 를 계산하므로, 텐서플로우에서 제공하는 학습된 VGG 19 불러와 저장해준다.
<br></br>

### SRGAN 학습하기

> SRGAN 학습
```python
from tensorflow.keras import losses, metrics, optimizers

generator = get_generator()
discriminator = get_discriminator()
vgg = get_feature_extractor()

# 사용할 loss function 및 optimizer 를 정의합니다.
bce = losses.BinaryCrossentropy(from_logits=False)
mse = losses.MeanSquaredError()
gene_opt = optimizers.Adam()
disc_opt = optimizers.Adam()

def get_gene_loss(fake_out):
    return bce(tf.ones_like(fake_out), fake_out)

def get_disc_loss(real_out, fake_out):
    return bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)


@tf.function
def get_content_loss(hr_real, hr_fake):
    hr_real = applications.vgg19.preprocess_input(hr_real)
    hr_fake = applications.vgg19.preprocess_input(hr_fake)
    
    hr_real_feature = vgg(hr_real) / 12.75
    hr_fake_feature = vgg(hr_fake) / 12.75
    return mse(hr_real_feature, hr_fake_feature)


@tf.function
def step(lr, hr_real):
    with tf.GradientTape() as gene_tape, tf.GradientTape() as disc_tape:
        hr_fake = generator(lr, training=True)
        
        real_out = discriminator(hr_real, training=True)
        fake_out = discriminator(hr_fake, training=True)
        
        perceptual_loss = get_content_loss(hr_real, hr_fake) + 1e-3 * get_gene_loss(fake_out)
        discriminator_loss = get_disc_loss(real_out, fake_out)
        
    gene_gradient = gene_tape.gradient(perceptual_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    
    gene_opt.apply_gradients(zip(gene_gradient, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
    return perceptual_loss, discriminator_loss


gene_losses = metrics.Mean()
disc_losses = metrics.Mean()

for epoch in range(1, 2):
    for i, (lr, hr) in enumerate(train):
        g_loss, d_loss = step(lr, hr)
        
        gene_losses.update_state(g_loss)
        disc_losses.update_state(d_loss)
        
        # 10회 반복마다 loss를 출력합니다.
        if (i+1) % 10 == 0:
            print(f"EPOCH[{epoch}] - STEP[{i+1}] \nGenerator_loss:{gene_losses.result():.4f} \nDiscriminator_loss:{disc_losses.result():.4f}", end="\n\n")
        
        if (i+1) == 200:
            break
            
    gene_losses.reset_states()
    disc_losses.reset_states()
```
SRGAN 의 학습에는 매우 오랜 시간이 소요되므로 처음부터 진행하지 않고, 200 번의 반복만 진행한다.

초반 학습이 불안정하여 Generator 의 loss 가 증가할 수 있다.
<br></br>

### SRGAN 테스트하기

SRGAN 은 만족스러운 결과를 도출하기까지 상당히 오랜 학습 시간을 요구한다.

때문에 미리 학습된 SRGAN 을 통해 테스트를 진행해보자.

SRGAN 은 생성자과 판별자로 구성되어 있지만 테스트에는 생성자만을 이용한다.
<br></br>
> 학습된 SRGAN 의 생성자 다운
```bash
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/srgan_G.h5 
$ mv srgan_G.h5 ~/aiffel/super_resolution
```
+ 참고 : [학습된 SRAN 모델 다운로드](https://aiffelstaticprd.blob.core.windows.net/media/documents/srgan_G.h5)
<br></br>
> 다운받은 SRGAN 모델 가져오기
```python
import tensorflow as tf
import os

model_file = os.getenv('HOME')+'/super_resolution/srgan_G.h5'
srgan = tf.keras.models.load_model(model_file)
```
<br></br>
> 테스트 과정을 진행하는 함수 생성 및 SRGAN 적용
```python
def apply_srgan(image):
    image = tf.cast(image[np.newaxis, ...], tf.float32)
    sr = srgan.predict(image)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)
    return np.array(sr)[0]

train, valid = tfds.load(
    "div2k/bicubic_x4", 
    split=["train","validation"],
    as_supervised=True
)

for i, (lr, hr) in enumerate(valid):
    if i == 6: break

srgan_hr = apply_srgan(lr)
```
이미지 전체를 시각화 했을 때 세부적인 선명함이 눈으로 판단하기 어렵다.
<br></br>
> 위에서 생성한 함수를 통한 이미지에 SRGAN 적용 결과 출력
> (일부 영역을 잘라내어 시각적으로 비교)
```python
# 자세히 시각화 하기 위해 3개 영역을 잘라냅니다.
# 아래는 잘라낸 부분의 좌상단 좌표 3개 입니다.
left_tops = [(400,500), (300,1200), (0,1000)]

images = []
for left_top in left_tops:
    img1 = crop(bicubic_hr, left_top, 200, 200)
    img2 = crop(srgan_hr , left_top, 200, 200)
    img3 = crop(hr, left_top, 200, 200)
    images.extend([img1, img2, img3])

labels = ["Bicubic", "SRGAN", "HR"] * 3

plt.figure(figsize=(18,18))
for i in range(9):
    plt.subplot(3,3,i+1) 
    plt.imshow(images[i])
    plt.title(labels[i], fontsize=30)
```
bicubic interpolation보다 훨씬 더 원래 고해상도 이미지에 가까운, 꽤나 만족할만한 결과를 얻을 수 있다.
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-19.srgan_exp.max-800x600.png)
<br></br>
위 그림은 SRGAN 논문에서 세부적으로 실험한 결과를 나타내며, 가장 첫 번째 SRResNet 은 SRGAN 의 Generator 를 뜻하며, Generator 구조만 이용해 SRCNN 과 비슷하게 MSE 손실함수로 학습한 결과이다.

오른쪽으로 갈수록 GAN 및 VGG 구조를 이용하여 점점 더 이미지 내 세부적인 구조가 선명해짐을 알 수 있다. (VGG 22 는 VGG 54 에 비해 더 low - level 의 특징에서 손실을 계산한다.)
<br></br>

## Super Resolution 결과 평가하기

Super Resolution 의 결과를 눈이 아닌 정량적으로 평가하는 여러 지표로는 PRNR, SSIM 등이 있다.
<br></br>
### PSNR 과 SSIM

PSNR (Peak Signal - to - Noise Ratio) 은 영상 내 신호가 가질 수 있는 최대 신호에 대한 잡음 (noise) 의 비율을 나타낸다.

일반적으로 영상을 압축했을 때, 화질이 얼마나 손실되는가 를 평가하는 목적으로 사용되며, 데시벨 (db) 단위를 사용한다.

PSNR 값이 높을수록 원본 영상에 비해 손실이 적다는 것을 의미한다.

SSIM (Structual Similarity Index Map) 은 영상의 구조 정보를 고려해 구조 정보를 얼마나 변화시키지 않았는지를 계산한다.

특정 영상에 대한 SSIM 값이 높을수록 원본 영상의 품질에 가깝다는 것을 의미한다.

+ 참고 : 
	+ [최대신호대잡음비(PSNR)와 이미지 품질](https://bskyvision.com/392)
	+ [2D 이미지 품질 평가에 구조변화를 반영하는 SSIM과 그의 변형들](https://bskyvision.com/396)
<br></br>

> 이미지 불러오기
```python
from skimage import data
import matplotlib.pyplot as plt

hr_cat = data.chelsea() # skimage에서 제공하는 예제 이미지를 불러옵니다.
hr_shape = hr_cat.shape[:2]

print(hr_cat.shape) # 이미지의 크기를 출력합니다.

plt.figure(figsize=(8,5))
plt.imshow(hr_cat)
```
<br></br>
> PSNR, SSIM 계산
> (scikit-image 라이브러리를 이용해 계산)
```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

print("**동일 이미지 비교**")
print("PSNR :", peak_signal_noise_ratio(hr_cat, hr_cat))
print("SSIM :", structural_similarity(hr_cat, hr_cat, multichannel=True))
```
`peak_signal_noise_ratio` , `structural_similarity` 두 메서드를 이용하면 쉽게 계산할 수 있다.

PSNR 과 SSIM 모두 높은 값을 가질수록 원본 이미지와 가깝다는 것을 의미하며, 동일한 이미지를 비교했기 때문에 두 결과는 각각 가질 수 있는 최댓값을 가진다.

PSNR 은 상한값이 없고, SSIM 은 0 ~ 1 사이의 값을 가지기 때문에 각각 inf 와 1 이 계산된다.
<br></br>
> 이미지의 가로 세로 픽셀수를 줄이고 bicubic interpolation을 이용해 원래 크기로 복원
```python
import cv2

# 이미지를 특정 크기로 줄이고 다시 늘리는 과정을 함수로 정의합니다.
def interpolation_xn(image, n):
    downsample = cv2.resize(
        image,
        dsize=(hr_shape[1]//n, hr_shape[0]//n)
    )
    upsample = cv2.resize(
        downsample,
        dsize=(hr_shape[1], hr_shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    return upsample

lr2_cat = interpolation_xn(hr_cat, 2) # 1/2로 줄이고 다시 복원
lr4_cat = interpolation_xn(hr_cat, 4) # 1/4로 줄이고 다시 복원
lr8_cat = interpolation_xn(hr_cat, 8) # 1/8로 줄이고 다시 복원

images = [hr_cat, lr2_cat, lr4_cat, lr8_cat]
titles = ["HR", "x2", "x4", "x8"]

# 각 이미지에 대해 PSNR을 계산하고 반올림합니다.
psnr = [round(peak_signal_noise_ratio(hr_cat, i), 3) for i in images]
# 각 이미지에 대해 SSIM을 계산하고 반올림합니다.
ssim = [round(structural_similarity(hr_cat, i, multichannel=True), 3) for i in images]

# 이미지 제목에 PSNR과 SSIM을 포함하여 시각화 합니다. 
plt.figure(figsize=(16,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i] + f" [{psnr[i]}/{ssim[i]}]", fontsize=20)
```
해상도를 줄일수록 그 이미지를 원래 크기로 interploation 했을 때, 각각의 계산 결과가 눈에 띄게 감소하는 것을 알 수 있다.
<br></br>

## SRCNN 및 SRGAN 결과 비교

> SRCNN 과 SRGAN 의 결과를 시각적으로만 비교
```python
for i, (lr, hr) in enumerate(valid):
    if i == 12: break # 12번째 이미지를 불러옵니다.

lr_img, hr_img = np.array(lr), np.array(hr)

# bicubic interpolation
bicubic_img = cv2.resize(
    lr_img, 
    (hr.shape[1], hr.shape[0]), 
    interpolation=cv2.INTER_CUBIC
)

# 전체 이미지를 시각화합니다.
plt.figure(figsize=(20,15))
plt.subplot(311); plt.imshow(hr_img)

# SRCNN을 이용해 고해상도로 변환합니다.
srcnn_img = apply_srcnn(bicubic_img)

# SRGAN을 이용해 고해상도로 변환합니다.
srgan_img = apply_srgan(lr_img)

images = [bicubic_img, srcnn_img, srgan_img, hr_img]
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

left_top = (700, 1090) # 잘라낼 부분의 왼쪽 상단 좌표를 지정합니다.

# bicubic, SRCNN, SRGAN 을 적용한 이미지와 원래의 고해상도 이미지를 시각화합니다.
plt.figure(figsize=(20,20))
for i, pind in enumerate([321, 322, 323, 324]):
    plt.subplot(pind)
    plt.imshow(crop(images[i], left_top, 200, 350))
    plt.title(titles[i], fontsize=30)
```
DIV2K 데이터셋 내에서 학습에 사용하지 않은 검증용 데이터셋을 이용하며, 몇 개 이미지만 뽑아서 Super Resolution 을 진행한 후 특정 부분을 잘라내어 확대하여 비교하였다.

Bicubic과 SRCNN은 많이 흐릿하지만 SRGAN의 결과는 매우 비슷함을 알 수 있다.
<br></br>
> 다른 이미지를 동일한 과정으로 비교
```python
for i, (lr, hr) in enumerate(valid):
    if i == 15: break

lr_img, hr_img = np.array(lr), np.array(hr)
bicubic_img = cv2.resize(
    lr_img, 
    (hr.shape[1], hr.shape[0]), 
    interpolation=cv2.INTER_CUBIC
)

plt.figure(figsize=(20,15))
plt.subplot(311); plt.imshow(hr_img)

srcnn_img = apply_srcnn(bicubic_img)
srgan_img = apply_srgan(lr_img)

images = [bicubic_img, srcnn_img, srgan_img, hr_img]
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

left_top = (600, 1500)

plt.figure(figsize=(20,20))
for i, pind in enumerate([321, 322, 323, 324]):
    plt.subplot(pind)
    plt.imshow(crop(images[i], left_top, 200, 350))
    plt.title(titles[i], fontsize=30)
```
<br></br>
```python
for i, (lr, hr) in enumerate(valid):
    if i == 8: break

lr_img, hr_img = np.array(lr), np.array(hr)
bicubic_img = cv2.resize(
    lr_img, 
    (hr.shape[1], hr.shape[0]), 
    interpolation=cv2.INTER_CUBIC
)

plt.figure(figsize=(20,15))
plt.subplot(311); plt.imshow(hr_img)

srcnn_img = apply_srcnn(bicubic_img)
srgan_img = apply_srgan(lr_img)

images = [bicubic_img, srcnn_img, srgan_img, hr_img]
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

left_top = (900, 1500)

plt.figure(figsize=(20,20))
for i, pind in enumerate([321, 322, 323, 324]):
    plt.subplot(pind)
    plt.imshow(crop(images[i], left_top, 200, 350))
    plt.title(titles[i], fontsize=30)
```
<br></br>
```python
for i, (lr, hr) in enumerate(valid):
    if i == 24: break

lr_img, hr_img = np.array(lr), np.array(hr)
bicubic_img = cv2.resize(
    lr_img, 
    (hr.shape[1], hr.shape[0]), 
    interpolation=cv2.INTER_CUBIC
)

plt.figure(figsize=(20,15))
plt.subplot(311); plt.imshow(hr_img)

srcnn_img = apply_srcnn(bicubic_img)
srgan_img = apply_srgan(lr_img)

images = [bicubic_img, srcnn_img, srgan_img, hr_img]
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

left_top = (700, 1300)

plt.figure(figsize=(20,20))
for i, pind in enumerate([321, 322, 323, 324]):
    plt.subplot(pind)
    plt.imshow(crop(images[i], left_top, 200, 350))
    plt.title(titles[i], fontsize=30)
```
<br></br>
```python
for i, (lr, hr) in enumerate(valid):
    # 불러올 이미지의 인덱스를 지정합니다.
    # 위에서 시각화 했던 8, 12, 15, 24 번을 제외한 다른 숫자를 넣어봅시다 
    if i ==32 : 
        break          

lr_img, hr_img = np.array(lr), np.array(hr)
bicubic_img = cv2.resize(
    lr_img, 
    (hr.shape[1], hr.shape[0]), 
    interpolation=cv2.INTER_CUBIC
)

plt.figure(figsize=(20,15))
plt.subplot(311); plt.imshow(hr_img)

srcnn_img = apply_srcnn(bicubic_img)
srgan_img = apply_srgan(lr_img)

images = [bicubic_img, srcnn_img, srgan_img, hr_img]
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

# 잘라낼 부분의 왼쪽 상단 좌표를 지정합니다.
left_top = (800, 1300)

plt.figure(figsize=(20,20)) # 이미지 크기를 조절할 수 있습니다.
for i, pind in enumerate([321, 322, 323, 324]):
    plt.subplot(pind)
    # crop 함수 내의 세번째 네번째 인자를 수정해 이미지 크기를 조절합니다.
    plt.imshow(crop(images[i], left_top, 200, 350))
    plt.title(titles[i], fontsize=30)
```
<br></br>
> Super Resolution 결과와 원래 고해상도 이미지 사이의 PSNR, SSIM 계산
```python
for i, (lr, hr) in enumerate(valid):
    if i == 24: break
    
lr_img, hr_img = np.array(lr), np.array(hr)
bicubic_img = cv2.resize(
    lr_img,
    (hr.shape[1], hr.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

srcnn_img = apply_srcnn(bicubic_img)
srgan_img = apply_srgan(lr_img)

images = [bicubic_img, srcnn_img, srgan_img, hr_img]
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

# 각 이미지에 대해 PSNR을 계산하고 반올림합니다.
psnr = [round(peak_signal_noise_ratio(hr_img, i), 3) for i in images]
# 각 이미지에 대해 SSIM을 계산하고 반올림합니다.
ssim = [round(structural_similarity(hr_img, i, multichannel=True), 3) for i in images]

# 이미지 제목에 PSNR과 SSIM을 포함하여 시각화 합니다. 
plt.figure(figsize=(18,13))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i] + f" [{psnr[i]}/{ssim[i]}]", fontsize=30)
```
제목 아래에 평가 결과를 표기되도록 하였으며, 시각적으로 가장 고해상도 이미지에 가까웠던 SRGAN 의 결과가 다른 방법들에 비해 낮은 PSNR 과 SSIM 결과를 가지는 것을 볼 수 있다.
<br></br>
> SRGAN 의 결과의 특정 부분을 잘라내고 잘라낸 부분의 PSNR, SSIM 을 계산
```python
left_top = (620, 570)
crop_images = [crop(i, left_top, 150, 250) for i in images]

psnr = [round(peak_signal_noise_ratio(crop_images[-1], i), 3) for i in crop_images]
ssim = [round(structural_similarity(crop_images[-1], i, multichannel=True), 3) for i in crop_images]

plt.figure(figsize=(18,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(crop_images[i])
    plt.title(titles[i] + f" [{psnr[i]}/{ssim[i]}]", fontsize=30)
```
SRCNN 결과의 경우 약간 선명해졌지만 전체적으로 여전히 고해상도 영상에 비해 흐릿하다.

이는 SRCNN 의 학습에 `Mean Squared Error` 를 사용했기 때문에, 생성해야 할 픽셀 값들을 고해상도 이미지와 비교해 단순히 평균적으로 잘 맞추는 방향으로 예측했기 때문이며, 이러한 문제는 SRCNN 뿐만 아니라 MSE 만을 사용해 학습하는 대부분의 신경망에서 발생하는 현상이기도하다.

SRGAN 결과의 경우 매우 선명하게 이상함을 확인할 수 있는데 Generator 가 고해상도 이미지를 생성하는 과정에서 Discriminator 를 속이기 위해 이미지를 진짜 같이 선명하게 만들도록 학습 되었기 때문이다.

또한 VGG 구조를 이용한 `content loss` 를 통해 학습한 것 또한 사실적인 이미지를 형성하는데 크게 기여했다. 다만, 입력되었던 저해상도 이미지가 매우 제한된 정보를 가지고 있기에 고해상도 이미지와 세부적으로 동일한 모양으로 선명하진 않은 것이다.
<br></br>
> 임의의 이미지를 골라내고 특정 영역에 대해서만 PSNR, SSIM 을 계산하여 제목 아래에 표기
```python
for i, (lr, hr) in enumerate(valid):
    # 불러올 이미지의 인덱스를 지정합니다.
    # 위에서 시각화 했던 8, 12, 15, 24 번을 제외한 다른 숫자를 넣어봅시다 
    if i == 3 : 
        break          

lr_img, hr_img = np.array(lr), np.array(hr)
bicubic_img = cv2.resize(
    lr_img,
    (hr.shape[1], hr.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

# 확대할 부분의 왼쪽 상단 좌표를 지정합니다.
left_top =(700, 1500) 

# crop 함수 내의 세번째 네번째 인자를 수정해 이미지 크기를 조절합니다.
crop_images = [crop(i, left_top, 150, 250) for i in images] 
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

psnr = [round(peak_signal_noise_ratio(crop_images[-1], i), 3) for i in crop_images]
ssim = [round(structural_similarity(crop_images[-1], i, multichannel=True), 3) for i in crop_images]

plt.figure(figsize=(18,10)) # 이미지 크기를 조절할 수 있습니다.
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(crop_images[i])
    plt.title(titles[i] + f" [{psnr[i]}/{ssim[i]}]", fontsize=30)
```
<br></br>

## 학습한 내용 정리하기

### 다른 이미지를 참고한다면?

지금까지 Super Resolution 방법은 세부적으로 SISR (Single Image Super Resolution) 에 해당한다.

즉, 한 장의 이미지를 이용해 고해상도 이미지를 생성하는 것이다.

+ 참고 : [Single Image Super Resolution using Deep Learning Overview](https://hoya012.github.io/blog/SIngle-Image-Super-Resolution-Overview/)
<br></br>

이때 여러 장의 이미지를 이용해 고해상도 이미지를 생성한다면 더 좋은 결과를 얻을 수 있을까?

이런 의문과 유사한 모델이 있는데 RefSR (Reference - based Super Resolution) 은 여러 장의 이미지를 이용해 고해상도 이미지를 생성하는 것이 아니라 해상도를 높이는데 참고할 만한 다른 이미지를 같이 제공하는 것이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-20.srntt_result.max-800x600.png)
<br></br>
위 그림은 RefSR 을 활용한 결과이며, `Ref images` 라고 나타난 두 이미지를 입력으로 Super Resolution을 수행했으며, `SRNTT` 라고 나타난 결과를 보면 이전에 학습했던 SRGAN 보다 훨씬 더 선명한 것을 확인할 수 있다.
<br></br>

### 차별을 재생성하는 인공지능

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-21.black_white_hands.max-800x600.png)
<br></br>
인공지능이 가진 문제들 중 하나는 차별이다. 인공지능 학습에서 사용하는 데이터가 편향되어있다면 이를 통해 학습된 인공지능 역시 편향될 것이다. 

이러한 예로 아마존이 개발한 인공지능 채용 / 인사 시스템이 여성에 대해 편견을 가지고 있다는 성 차별적 사례가 있었으며, 구글의 이미지 인식 모델이 위 그림과 같이 손의 피부색에 따라 체온계를 총으로 인식하는 인종차별 사례가 있었다.

이러한 차별의 문제는 Super Resolution 에도 존재한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-22.pulse_result.max-800x600.png)
<br></br>
위 그림은 2020 년 초에 발표된 PULSE 라는 구조를 활용한 Super Resolution 결과물로 성능은 매우 뛰어나다. 가장 왼쪽 그림과 같이 거의 정보가 존재하지 않는 저해상도에서 정보를 새롭게 생성하는 수준을 보여준다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-22-23.pulse_bad_result.max-800x600.png)
<br></br>
하지만 이러한 모델에 저해상도 얼굴 이미지를 입력하면 모두 백인으로 만들어 내는 문제가 있었다.

위 그림은 오바바 전 미국 대통력의 이미지와, 저해상도의 동양인 혹은 캐릭터를 입력으로 넣을때에도 백인으로 생성됨을 알 수 있다.

이는 PULSE 모델이 백인 위주로 구성된 데이터로 학습된 것으로 기인했다고 연구자들은 말하고 있다.

백인으로 편향된 학습 데이터 때문에 새롭게 생성된 픽셀들이 기본적으로 흰색에 가깝게 설정되거나 백인의 이목구비를 나타내는 것이다.

+ 참고 : [공정한 AI 얼굴인식기](https://www.kakaobrain.com/blog/57)
<br></br>
