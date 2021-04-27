# 01. 인공지능과 가위바위보 하기

## 인공지능과 가위바위보 하기

### 간단한 이미지 분류기

목표 : 이미지를 분류하는 간단한 인공지능 모델 구현 (숫자 손글씨 인식, 가위바위보 게임)

### 숫자 손글씨 인식기 만들기 (Sequential Model 이용)

손으로 쓴 숫자 이미지를 입력으로 받아 그 이미지가 무슨 숫자를 나타내는지를 출력한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-1-3.max-800x600.jpg)

## How to make?

일반적으로 딥러닝 기술은 

1. __데이터 준비__
2. __딥러닝 네트워크 설계__
3. __학습__ 
4. __테스트 (평가)__

의 과정을 따르게 된다.

## 데이터 준비

### MNIST 숫자 손글씨 Dataset 불러오기

- 사용할 데이터 : __[MNIST](http://yann.lecun.com/exdb/mnist/)__
![[숫자 손글씨 데이터들 ( 그림 출처 : https://commons.wikimedia.org/wiki/File:MnistExamples.png )]](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-1-1.png)



텐서플로우의 표준 API 인 `tf.keras` 의 Sequential API 를 이용하여 이미지 분류모델을 구현한다. (텐서플로우 2.0 이상의 버전 이용)


> MNIST 데이터셋 읽어오기
```python
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)   # Tensorflow의 버전을 출력

mnist = keras.datasets.mnist

# MNIST 데이터를 로드. 다운로드하지 않았다면 다운로드까지 자동으로 진행됩니다. 
(x_train, y_train), (x_test, y_test) = mnist.load_data()   

print(len(x_train))  # x_train 배열의 크기를 출력
```
<br></br>
> MNIST 의 손글씨 이미지 출력

`x_train` 에 담긴 2번째 이미지 확인
```python
plt.imshow(x_train[1],cmap=plt.cm.binary)
plt.show()
```
`y_train` 에 담긴 2번째 이미지 확인
```python
print(y_train[1])
```
다른 임의의 이미지 확인
```python
# index에 0에서 59999 사이 숫자를 지정해 보세요.
index=10000     
plt.imshow(x_train[index],cmap=plt.cm.binary)
plt.show()
print( (index+1), '번째 이미지의 숫자는 바로 ',  y_train[index], '입니다.')
```
* [Matplotlib](https://matplotlib.org/gallery.html) : 시각화 패키지로 다양한 형태로 시각화 할 수 있는 기능을 제공

### 학습용 데이터와 시험용 데이터

> 학습용 데이터의 갯수 확인
```python
print(x_train.shape)
```
<br></br>
> 테스트용 데이터 갯수 확인
```python
print(x_test.shape)
```
<br></br>
[참고 : 데이터셋의 구분](https://tykimos.github.io/2017/03/25/Dataset_and_Fit_Talk/)
데이터셋은 학습용 데이터와 테스트용 데이터, 검증용 데이터로 구분된다. 

- 학습용 데이터 : 데이터를 학습하는데 사용되는 데이터
- 테스트용 데이터 : 학습된 모델을 평가하는데 사용되는 데이터
- 검증용 데이터 : 트레이닝 셋으로 학습 한 후 테스트 셋 전, 즉 트레이닝 셋과 테스트 셋 사이에 벨리데이션 셋을 사용.  즉, 정상적으로 학습이 되고 있는지 판단할 때 입니다. 언더피팅상태인지,  오버피팅이 발생했는지, 학습을 중단해도 될지를 판단할 때 사용

### 데이터 전처리

숫자 손글씨 이미지의 실제 픽셀값은 0~255 사이의 값을 가지며, 이 이미지를 컴퓨터가 인식할 수 있게 해주기 위해 숫자로 변환해 줄 필요가 있다.

> 숫자 손글씨 이미지의 픽셀값 확인
```python
print('최소값:',np.min(x_train), ' 최대값:',np.max(x_train))
```
<br></br>
인공지능 모델 훈련 시 일반적으로 입력을 0~1 사이의 값으로 정규화 시켜줘야한다.
<br></br>
> 숫자 손글씨 이미지 픽셀값을 정규화
> 0~255 사이의 값을 가지므로 255.0 으로 나눠준다.
```python
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0
print('최소값:',np.min(x_train_norm), ' 최대값:',np.max(x_train_norm))
```

## 딥러닝 네트워크 설계

### Sequential Model 을 사용한 딥러닝 네트워크 설계

텐서플로우 케라스 `tf.keras` 에서 제공하는 Sequentaial API 를 사용한다. 

개발의 자유도는 떨어지지만 간단한게 딥러닝 모델을 만들 수 있다.

케라스를 통한 모델 설계 방법은 Sequential API, Functional API, 밑바닥부터 직접 코딩하는 방법 등 이 있다.
<br></br>
> `tf,keras` 의 Sequential API 를 통해 LeNet 라는 딥러닝 네트워크를 설계
```python
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

print('Model에 추가된 Layer 개수: ', len(model.layers))
```
<br></br>
![코드의 의미](https://aiffelstaticprd.blob.core.windows.net/media/images/F-1-5.max-800x600.png)

+ Conv2D 레이어의 첫번째 인자 : 사용하는 이미지 특징의 수. 16개의 이미지 특징과 32개의 이미지 특징을 고려한다. (숫자보다 더욱 복잡한 이미지라면 특징 숫자를 늘려주는 것을 고려한다.)

+ Dense 레이어의 첫 번째 인자 : 분류기에 사용되는 뉴런의 숫자. (값이 클수록 복잡한 분류기를 만들 수 있다. 알파벳을 구분하고자 할 때는 총 52개 (대문자 25개, 소문자 26개) 가 필요하다 따라서 32보다 큰 64, 128 등을 고려해 볼 수 있다.)

+ 마지막 Dense 레이어의 뉴런 숫자 : 분류해야하는 클래스의 수. (숫자 이미지 분류는 0~9 까지의 10개 숫자이기에 10이 된다.)
<br></br>
> 구현한 딥러닝 네트워크 모델 확인
```python
model.summary()
```
![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-1-6.max-800x600.png)


## 딥러닝 네트워크 학습

생성한 딥러닝 네트워크는 `(데이터 갯, 이미지 크기 x, 이미지 크기 y, 채널수)` 형태를 가진다.

하지만 `print(x_train.shape)` 를 통해 형태를 확인 해 보면 `(60000, 28, 28)` 로 채널수에 대한 정보가 없다. 따라서 `(60000, 28, 28, 1)` 과 같이 채널수에 대한 정보를 가지도록 변경해주어야 한다.

+ 채널수는 마지막 1 에 해당하며, 컬러의 경우 3, 흑백의 경우 1 을 의미한다.
<br></br>
> 학습용 데이터와 테스트용 데이터의 형태를 확인 및 변경
```python
print("Before Reshape - x_train_norm shape: {}".format(x_train_norm.shape))
print("Before Reshape - x_test_norm shape: {}".format(x_test_norm.shape))

x_train_reshaped=x_train_norm.reshape( -1, 28, 28, 1)  # 데이터갯수에 -1을 쓰면 reshape시 자동계산됩니다.
x_test_reshaped=x_test_norm.reshape( -1, 28, 28, 1)

print("After Reshape - x_train_reshaped shape: {}".format(x_train_reshaped.shape))
print("After Reshape - x_test_reshaped shape: {}".format(x_test_reshaped.shape))
```
<br></br>
> 학습 데이터를 통한 딥러닝 네트워크 학습
```python
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train_reshaped, y_train, epochs=10)
```
학습 데이터로는 딥러닝 모델에 맞게 형태를 변경해준 `X_train_reshaped` 를 사용한다.

모델 학습시 진행에 따라 epoch 별 인석 정확도 `accuracy` 를 확인할 수 있다. 이를 통해 적절한 epoch 정도를 정한다. 

+ 상승률이 거의 없다면 그 이상의 epoch 는 불필요하다.

## 얼마나 잘 만들었는지 확인

### 테스트 데이터로 성능 확인

`X_train`, 즉 학습용 데이터를 통한 정확도는 연습문제를 풀어본 것과 같다.

따라서 실제 시험을 통해 잘 학습했는지를 판단해 볼 필요가 있다.

이 실제 시험에 해당하는 것이 바로 `x_test`, 즉 테스트 데이터이다.
<br></br>
> 테스트 데이터를 통한 성능 평가
```python
test_loss, test_accuracy = model.evaluate(x_test_reshaped,y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
```
<br></br>
학습용 데이터의 정확도 보다 소폭 하락한 것을 볼 수 있다.

학습 시 한 번도 본적없는 필체의 손글씨가 테스트 데이터에 있었을 가능성이 있으며, 이는 예상 가능한 결과이다.

## 어떤 데이터를 잘못 추론했을까? 눈으로 확인해보자

`model.evaluate()` 대신 `model.predict()` 를 사용하면 모델이 입력값을 통해 실제 추론한 확률분포를 출력할 수 있다.
<br></br>
> 모델이 출력한 확률값 확인
```python
predicted_result = model.predict(x_test_reshaped)  # model이 추론한 확률값. 
predicted_labels = np.argmax(predicted_result, axis=1)

idx=0  #1번째 x_test를 살펴보자. 
print('model.predict() 결과 : ', predicted_result[idx])
print('model이 추론한 가장 가능성이 높은 결과 : ', predicted_labels[idx])
print('실제 데이터의 라벨 : ', y_test[idx])
```
<br></br>
출력 결과는 `[9.5208375e-15 2.8931768e-11 1.2696462e-09 2.0265421e-08 6.1321614e-11 2.9599554e-12 1.0710074e-15 1.0000000e+00 1.0549885e-11 3.8589491e-08]` 의 백터 형태로 출력되며, 0~9 까지의 숫자로 인식했을 확률을 의미한다.

즉, 1에 가까울 수록 추론에 확신하고 있다는 의미를 가진다.
<br></br>
> 추론한 숫자와 실제 정답 숫자를 확인
```python
plt.imshow(x_test[idx],cmap=plt.cm.binary)
plt.show()
```
<br></br>
> 모델이 추론한 숫자와 실제 숫자가 다른 경우를 시각화
```python
import random
wrong_predict_list=[]
for i, _ in enumerate(predicted_labels):
    # i번째 test_labels과 y_test이 다른 경우만 모아 봅시다. 
    if predicted_labels[i] != y_test[i]:
        wrong_predict_list.append(i)

# wrong_predict_list 에서 랜덤하게 5개만 뽑아봅시다.
samples = random.choices(population=wrong_predict_list, k=5)

for n in samples:
    print("예측확률분포: " + str(predicted_result[n]))
    print("라벨: " + str(y_test[n]) + ", 예측결과: " + str(predicted_labels[n]))
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()
```
<br></br>
 틀린 경우 추론에 대한 확신하는 정도가 낮다는 것을 알 수 있다. 이렇게 시각화를 통해 향후 모델 성능 개선 방법에 대한 아이디어를 얻을 수 있다.


## 더 좋은 네트워크 만들어 보기

모델의 성능을 개선하는 방법 중 가장 보편적인 방법은 __하이퍼파라미터__ 를 바꿔보는 것이다.

즉, `Conv2D` 레이어에서 입력 이미지의 특징 수를 늘리거나 줄여 보거나, `Dense` 레이어에서 뉴런수를 바꾸어 보거나, 학습 반복 횟수인 `epoch` 값을 변경해 볼 수 있다.
<br></br>
> 하이퍼파라미터 변경을 통한 모델 개선
```python
#바꿔 볼 수 있는 하이퍼파라미터들
n_channel_1=25
n_channel_2=40
n_dense=32
n_train_epoch=10

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# 모델 훈련
model.fit(x_train_reshaped, y_train, epochs=n_train_epoch)

# 모델 시험
test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
```

