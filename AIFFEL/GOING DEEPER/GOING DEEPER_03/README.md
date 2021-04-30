# 03. 잘 만든 Augmentation, 이미지 100장 안 부럽다.

모델 학습에는 많은 양의 데이터가 필요하다. 하지만 많은 양의 데이터를 모으는데는 엄청난 자원이 소요된다.

특히 컴퓨터 비전 분야에 사용될 데이터는 주로 이미지인데, 이러한 제한된 데이터셋을 최대한 활용하는 방법으로 Augmentation 이라는 방법이 존재한다.


## 실습 목표

1.  Augmentation을 하는 이유를 알아갑니다.
2.  여러 가지 Augmentation 방법을 알아둡니다.
3.  학습에 Augmentation을 적용할때 주의해야 할 점을 숙지합니다.


## 학습내용

1.  데이터셋의 현실
2.  Data Augmentation이란?
3.  텐서플로우를 사용한 Image Augmentation
4.  imgaug 라이브러리
5.  더 나아간 기법들


## 준비

> 학습 경로 설정
```python
$ mkdir -p ~/aiffel/data_augmentation/images
```
<br></br>

## 데이터셋의 현실

### 대량의 데이터셋

이미지넷은 1,400 만장의 이미지를 보유하고 있으며, CIFAR - 10 의 학습용 이미지는 5만장이다.

이렇게 많은 이미지를 직접 구축하기 위해서는 기상천외한 양의 자원이 소요된다.

<br></br>
### 직접 구축하는 데이터셋

이미지 데이터를 구축하기 위해서 쉽게 크롤링을 떠올릴 수 있다. 하지만 목적에 맞는 이미지를 위해 불필요한 부분을 제거하고나면 많은 양의 이미지가 제거될 것이다.

또한 수 만장의 이미지를 데이터셋으로 구축한다 하더라도 이를 고품질로 정제하는 과정 역시 무시할 수 없다.

<br></br>
## Data Augmentation 이란? (01). 개요

보유한, 혹은 수집한 데이터셋을 최대한 활용하기 위한 방법으로 Data Augmentation 이라는 방법을 사용한다.

이름에서도 유추할 수 있듯 데이터를 증강 시켜 학습 데이터셋의 규모를 키우는 방법이다.

* 참고 : [Data Augmentation](https://youtu.be/JI8saFjK84o/0.jpg)

데이터셋이 실제 상황에서의 입력값과 다를 경우, augmentation 을 통해 실제 입력값과 비슷한 분포를 만들어 낼 수 있다.

예를 들면 실제 입력 이미지에서 노이즈가 많을 경우 동일 클래스의 이미지로 학습했다 하더라도 다른 결과를 출력할 수 있는데, augmentation 은 데이터셋의 개수를 증강시킬 뿐만아니라 이런 실제 환경에서 발생할 수 있는 문제들을 고려한 이미지를 만들도록 도와줄 수 있다.

<br></br>
### 이미지 데이터 Augmentation

실제로 이미지 데이터 Augmentation 은 포토샵이나 사진이 필터 등에서 사용하는 기술과 매우 흡사하다.

즉, 한 장의 이미지를 가지고 좌우 대칭을 만들거나, 색상을 조절하는 등의 역할을 하는 것이다.


<br></br>
## Data Augmentation (02). 다양한 Image Augmentation 방법

텐서플로우 튜토리얼에는 Image Augmentation 의 다양한 예제를 소개함과 동시에 API 를 사용해 적용할 수 있는 Image Augmentation 기법을 제공한다.

* 참고 : [Tensorflow의 data_augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

<br></br>
### Flipping

![](https://aiffelstaticprd.blob.core.windows.net/media/images/gc-2-l-flip.max-800x600.png)

Flip 은 핸드폰 이름에서도 알 수 있듯이 상하로 이미지를 반전시키는 방법이다.

이 Flipping 기능을 적용할 때에는 주의할 점이 있다.

이미지 분류 문제를 해결할 때 적용한다면 큰 문제가 없을 수 있다. 하지만 물체 탐지 (object detection), 세그멘테이션 (segmentation) 문제, 인식 (recognition) 등 정확한 정답 영역이 존재하는 문제에 적용할 때에는 정답 영역이 있는 라벨에도 같이 좌우 반전을 적용해 주어야한다.

<br></br>
### Gray Scale

![](https://aiffelstaticprd.blob.core.windows.net/media/images/gc-2-l-grayscale.max-800x600.png)

Gray Scale 은 흑백 채널을 의미한다. 즉, RGB 3개의 채널을 가지는 컬러 이미지를 흑백 이미지로 변환해 준다.

위 그림에서는 흑백이 아닌 G 색상으로 표현하였지만 원리는 같다.

RGB 각각의 채널마다 가중치 (weight) 를 주어 가중합 (weighted sum) 을 구하는 방식으로 구현하며, 가중치의 경우 합이 1 이 된다.

<br></br>
### Saturation

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-2-L-5.max-800x600.png)

Saturation 은 RGB 이미지를 HSV (Hue (색조), Saturation (채도), Value (명도) 3 가지 성분으로 색을 표현) 이미지로 변경하고 S (Saturation) 채널에 오프셋 (offset) 을 적용하여 이미지를 보다 선명하게 만들어준다.

<br></br>
### Brightness

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-2-L-6.max-800x600.png)

이름에서도 알 수 있 듯 밝기를 조절하는 역할을 한다.

RGB 이미지에서 (255, 255, 255) 는 흰색을 의미하며, (0, 0, 0) 은 검은색을 의미한다.

즉 값을 더해줄 수록 이미지가 밝아지고, 값을 뺄 수록 이미지가 어두워진다. 이를 통해 밝기를 조절할 수 있다.

<br></br>
### Rotation

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-2-L-7.max-800x600.png)

Rotation 은 이미지의 각도를 변환해준다.

90 도 로 변환할 경우 직사각형의 형태가 유지되므로 이미지의 크기만 가로 세로를 세로, 가로로 변경해주면 사용할 수 있다. 

하지만 90 도 외에 다른 각도로 변환할 경우 필연적으로 이미지가 잘리는 영역이 발생하게되며, 이렇게 기존 이미지로 채우지 못하는 영역을 어떻게 처리해야 할 지 고민해야한다.

<br></br>
### Center Crop

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-2-L-8.max-800x600.png)

Center Crop 는 이미지 중앙을 기준으로 확대하는 방법이다.

너무 작게 Center Crop 을 할 경우 본래 이미지와는 다른 이미지로 변경되어 라벨과 일치하지 않을 수 있으므로 주의해야 한다. 

(예를 들어 사람의 얼굴을 너무 확대하여 피부의 일부만 보이거나 동물을 너무 확대하여 털 만 보이는 경우)

<br></br>
### 기타 Agumentation 방법들

Gaussian noise, Contrast change, Sharpen,  Affine transformation, Padding, Blurring 등 여러 Augmentation 방법이 존재한다.

Augmentation 코드는 직접 만들거나 새로운 라이브러리를 활용해야한다.

<br></br>
## 텐서플로우를 사용한 Image Augmentation (01). Flip

Agumentation 의 여러 기법을 코드로 적용해보자.

먼저 이미지를 다루게 되므로, PIL  (pillow) 라이브러리를 활용한다.

> PIL 라이브러리 설치
```python
pip install pillow
```
<br></br>
> 고양이 사진을 다운받아 작업 디렉토리인 images 폴더로 이동
* [고양이 이미지](https://aiffelstaticprd.blob.core.windows.net/media/documents/mycat.jpg)
```
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/mycat.jpg 
$ mv mycat.jpg ~/aiffel/data_augmentation/images
```
<br></br>
> 라이브러리 임포트 및 이미지 저장경로 설정
```python
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os

sample_img_path = os.getenv('HOME')+'/aiffel/data_augmentation/images/mycat.jpg'
sample_img_path
```
<br></br>
> 이미지 리사이징
```python
image = Image.open(sample_img_path).resize((500, 400)) # 이미지에 따라 숫자를 바꾸어 보세요.
image_tensor = tf.keras.preprocessing.image.img_to_array(image)

image
```
<br></br>
> Flip 적용
> (Flip 는 `flip_left_right`, `flip_up_down` 2가지로 구분된다.)
```python
flip_lr_tensor = tf.image.flip_left_right(image_tensor)
flip_ud_tensor = tf.image.flip_up_down(image_tensor)
flip_lr_image = tf.keras.preprocessing.image.array_to_img(flip_lr_tensor)
flip_ud_image = tf.keras.preprocessing.image.array_to_img(flip_ud_tensor)

plt.figure(figsize=(15, 15))

plt.subplot(1,3,1)
plt.title('Original image')
plt.imshow(image)

plt.subplot(1,3,2)
plt.title('flip_left_right')
plt.imshow(flip_lr_image)

plt.subplot(1,3,3)
plt.title('flip_up_down')
plt.imshow(flip_ud_image)
```
일괄적으로 상화, 좌우 반전을 적용한다.
<br></br>
> 확률에 따른 Flip 적용
> (`random_flip_left_right`, `random_flip_up_down` 사용)
```python
plt.figure(figsize=(12, 16))

row = 4
for i in range(row):
    flip_lr_tensor = tf.image.random_flip_left_right(image_tensor)
    flip_ud_tensor = tf.image.random_flip_up_down(image_tensor)
    flip_lr_image = tf.keras.preprocessing.image.array_to_img(flip_lr_tensor)
    flip_ud_image = tf.keras.preprocessing.image.array_to_img(flip_ud_tensor)
    
    plt.subplot(4,3,i*3+1)
    plt.title('Original image')
    plt.imshow(image)

    plt.subplot(4,3,i*3+2)
    plt.title('flip_left_right')
    plt.imshow(flip_lr_image)

    plt.subplot(4,3,i*3+3)
    plt.title('flip_up_down')
    plt.imshow(flip_ud_image)
```
확률에 따라 Flip 을 적용되도록 해야 원본 데이터도 활용이 가능하다.`random_flip` 로 상하좌우 반전을 적용하는 함수도 생성이 가능하다.
<br></br>

## 텐서플로우를 사용한 Image Augmentation (02). Center Crop

> Center Crop 적용
```python
plt.figure(figsize=(12, 15))

central_fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
col = len(central_fractions)
for i, frac in enumerate(central_fractions):
    cropped_tensor = tf.image.central_crop(image_tensor, frac)
    cropped_img = tf.keras.preprocessing.image.array_to_img(cropped_tensor)
    
    plt.subplot(1,col+1,i+1)
    plt.title(f'Center crop: {frac}')
    plt.imshow(cropped_img)
```
`central_fraction` 은 얼마나 확대 할 지를 조절하는 매개변수이다. 1.0 은 원본 이미지이며, 숫자가 작아질 수록 확대된다.
<br></br>
> 랜덤하게 Center Crop 을 적용하는 함수 구현
```python
def random_central_crop(image_tensor, range=(0, 1)):
    central_fraction = tf.random.uniform([1], minval=range[0], maxval=range[1], dtype=tf.float32)
    cropped_tensor = tf.image.central_crop(image_tensor, central_fraction)
    return cropped_tensor
```
기본적으로 Flip 과 달리 랜덤하게 적용하는 함수는 제공되지 않는다. 

따라서 파이썬의 `random` 모듈을 사용하거나 텐서플로우의 `tf.random.uniform` 랜덤 모듈을 사용한다.

`central_fraction` 매개변수에 전달할 값을 만들고, 이를 활용하여 `cropped_tensor` 를 만들어 내는 `random_central_crop` 함수를 위에서 구현하였다.
<br></br>
> 랜덤 Center Crop 적용
```python
plt.figure(figsize=(12, 15))

col = 5
for i, frac in enumerate(central_fractions):
    cropped_tensor =random_central_crop(image_tensor)
    cropped_img = tf.keras.preprocessing.image.array_to_img(cropped_tensor)
    
    plt.subplot(1,col+1,i+1)
    plt.imshow(cropped_img)
```
<br></br>

## 텐서플로우를 사용한 Image Augmentation (03). 직접 해보기

`tf.image.random_crop()` 와 `tf.image.random_brightness()` 를 구현해보자.

* 참고 : 
	* [tf.image.random_crop](https://www.tensorflow.org/api_docs/python/tf/image/random_crop)
	* [f.image.random_brightness](https://www.tensorflow.org/api_docs/python/tf/image/random_brightness)
<br></br>
> `random_crop()` 을 적용
```python
# apply random_crop on cat image
plt.figure(figsize=(12, 15))

random_crop_tensor = tf.image.random_crop(image_tensor,[180,180,3])
random_crop_image = tf.keras.preprocessing.image.array_to_img(random_crop_tensor)

plt.subplot(1,3,1)
plt.imshow(random_crop_image)
```
<br></br>
> 적용한 `random_crop()` 을 시각화
```python
# display 5 random cropped images
plt.figure(figsize=(12, 15))

for i in range(5):
  random_crop_tensor = tf.image.random_crop(image_tensor,[200,200,3])
  random_crop_image = tf.keras.preprocessing.image.array_to_img(random_crop_tensor)
  plt.subplot(1,5,i+1)
  plt.imshow(random_crop_image)
```
<br></br>
> `tf.image.random_brightness()` 적용
```python
# apply random_brightness on cat image

cropped_tensor = tf.image.random_brightness(image_tensor, max_delta=255)
plt.imshow(cropped_img)
```
`tf.image.random_brightness()`만 적용할 경우 이미지 텐서 값의 범위가 0~255를 초과하게 될 수도 있ek.

이 경우 `plt.imshow()`에서 rescale되어 밝기 변경 효과가 상쇄되어 보일 수도 있다.

`tf.image.random_brightness()` 다음에는 `tf.clip_by_value()`를 적용해 주는 것 명심하자.
<br></br>
> 적용한 `tf.image.random_brightness()` 시각화
```python
# display 5 random brightness images

plt.figure(figsize=(12, 15))

col = 5
for i in range(5):
    cropped_tensor = tf.image.random_brightness(image_tensor, max_delta=255)
    cropped_img = tf.keras.preprocessing.image.array_to_img(cropped_tensor)

    plt.subplot(1,col+1,i+1)
    plt.imshow(cropped_img)
```
<br></br>

## Imgaug 라이브러리

### Imgaug 라이브러리 사용하기

Imgaug 라이브러리는 Augmentation 만을 모아서 제공하는 전문 라이브러리이다.

앞서 말했듯이 augmentation을 적용할 때는 학습용 데이터 외에 정답(ground truth 또는 gt)이 되는 데이터에도 augmentation 이 동일하게 적용되어야 한다.

* 참고 :
	 - [imgaug 라이브러리](https://github.com/aleju/imgaug)
	 - [Overview of imgaug](https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html)
<br></br>
> Imgaug 라이브러리 설치
```python
pip install -q imgaug
```
`imgaug` 는 배열을 이미지의 기본 형태로 사용한다. 

때문에 PIL Image 데이터형을 넘파이 배열로 변환하여 사용해야 한다.
<br></br>
> 넘파이 임포트 및 이미지를 넘파이 배열로 변환
```python
import numpy as np
import imgaug.augmenters as iaa

image_arr = np.array(image)
```
<br></br>

### Augmentation 기법 사용해 보기

1. iaa.Affine()

`imgaug.augmenters` 의 `Affine()` 은 아핀 변환 (Affine transform) 을 이미지에 적용한다.

아핀 변환이란 2D 변환의 일종으로 이미지의 스케일 (scale) 을 조절하거나 평행이동, 회전등의 변환을 줄 수 있다.

+ 참고 : [2D 변환](https://darkpgmr.tistory.com/79)

> 아핀 변환 `iaa.Affine()` 적용
```python
images = [image_arr, image_arr, image_arr, image_arr]
rotate = iaa.Affine(rotate=(-25, 25))
images_aug = rotate(images=images)
plt.figure(figsize=(13,13))
plt.imshow(np.hstack(images_aug))
```
-25 ~ 25 사이로 랜덤하게 각도를 변환하는 augmentation 을 볼 수 있다.
<br></br>

2. iaa.Crop()

텐서플로우 API 를 활용한 Crop 과 동일하다.

원본 이미지의 비율을 매개변수로 사용하여 이미지를 생성한다.
> iaa.Crop() 적용
```python
images = [image_arr, image_arr, image_arr, image_arr]
crop = iaa.Crop(percent=(0, 0.2))
images_aug = crop(images=images)
plt.figure(figsize=(13,13))
plt.imshow(np.hstack(images_aug))
```
<br></br>

3. iaa.Sequential()

`iaa.Sequential` 은 여러 augmentation 기법들을 정해진 순서대로 차례로 적용할 수 있는 기법이다.

> iaa.Sequential 적용
```python
images = [image_arr, image_arr, image_arr, image_arr]
rotate_crop = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.Crop(percent=(0, 0.2))
])
images_aug = rotate_crop(images=images)
plt.figure(figsize=(13,13))
plt.imshow(np.hstack(images_aug))
```
rotate 와 crop 기법을 순차적으로 적용한 것을 볼 수 있다.
<br></br>
> `iaa.Sequential()` 의 augmentation 순서를 random 하게 바꿔 사용
```python
# modify iaa.sequential to use random step

images = [image_arr, image_arr, image_arr, image_arr]
rotate_crop = iaa.Sequential([
    iaa.Crop(percent=(0, 0.2)),
    iaa.Affine(rotate=(-25, 25)),
], random_order=True)
images_aug = rotate_crop(images=images)
plt.imshow(np.hstack(images_aug))
```
기본적으로 augmentation 순서는 고정이다. 

하지만 random 모듈을 활용하여 이 순서를 랜덤하게 바꿀 수 있다.
<br></br>

4. iaa.OneOf()

여러 augmentation 중 하나만 선택하여 적용하는 방법.
> iaa.OneOf() 적용
```python
images = [image_arr, image_arr, image_arr, image_arr]
seq = iaa.OneOf([
     iaa.Grayscale(alpha=(0.0, 1.0)),
     iaa.AddToSaturation((-50, 50))
])
images_aug = seq(images=images)
plt.figure(figsize=(13,13))
plt.imshow(np.hstack(images_aug))
```
<br></br>

5. iaa.Sometimes()

여러 augmentation 들이 일정 확률로 선택되는 기능.

iaa.OneOf() 기능과 유사하다.

* 참고 : [iaa.Sometimes()](https://imgaug.readthedocs.io/en/latest/source/overview/meta.html#sometimes)

<br></br>
> iaa.Sometimes() 적용
```python
# Use iaa.SomeTimes with AddToSaturation & Grayscale

images = [image_arr, image_arr, image_arr, image_arr]
seq = iaa.Sequential([
     iaa.Sometimes(
         0.6,
         iaa.AddToSaturation((-50, 50))
     ),
     iaa.Sometimes(
         0.2,
         iaa.Grayscale(alpha=(0.0, 1.0))
     )
])
images_aug = seq(images=images)
plt.imshow(np.hstack(images_aug))
```
<br></br>
> 개와 고양이를 분류하는 모델을 만든다는 가정하에 augmentation 적용
> (1024 가지가 넘는 augmentation 을 구현, 그 중 100장만 시각화)
```python
# Use various techniques and functions in imgaug library. Make at least 1,024 images and show 100 images.

seq = iaa.Sequential([
    iaa.OneOf([
         iaa.Grayscale(alpha=(0.0, 1.0)),
         iaa.Sometimes(
             0.5,
             iaa.AddToSaturation((-50, 50))
         )
    ]),
    iaa.Sequential([
        iaa.Crop(percent=(0, 0.2)),
        iaa.Affine(rotate=(-25, 25)),
    ], random_order=True)
])

plt.figure(figsize=(10, 40))
for i in range(20):
    images = [image_arr, image_arr, image_arr, image_arr, image_arr]
    images_aug = seq(images=images)
    plt.subplot(20,1,i+1)
    plt.imshow(np.hstack(images_aug))

plt.show()
```
<br></br>

## 더 나아간 기법들

대표적으로 augmentation 이 활용되는 분야는 GAN 이다.

논문에는 전통적인 augmentation 방법과 GAN 을 활용한 augmentation 을 적용하여 효과를 비교, 실험하였다.

대표적인 예로는 스타일 트랜스퍼 (style transfer) 모델이 있다.

+ 참고 : [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)  _by Jason Wang and Luis Perez_
