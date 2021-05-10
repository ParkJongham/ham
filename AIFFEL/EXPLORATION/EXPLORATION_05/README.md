# 05. 사람이 말하는 단어를 인공지능 모델로 구분해보자


## 학습 목표

-   Audio 형태의 데이터를 다루는 방법에 대해서 알아보기
-   Wav 파일의 형태와 원리를 이해하기
-   오디오데이터를 다른 다양한 형태로 변형시켜보기
-   차원이 다른 데이터에 사용가능한 classification 모델 직접 제작해보기

<br></br>
## 음성과 오디오 데이터

### 파동으로서의 소리

파동은 진동으로 인한 소리의 압축이 얼마나 되었는지를 설명한다. 즉, 소리를 설명할 수 있는 것이 파동이다.

소리는 3가지 물리량 진폭 (Amplitude), 주파수 (Frequency), 위상 (Phase) 으로 나타낼 수 있다.

물리적 음향에서 강도 (Intensity) 는 진폭의 세기로 정의되며, 주파수는 떨림의 빠르기, Tone - Color 는 파동의 모양을 정의한다.

심리적 음향에서 Loudness 는 소리의 크기를 의미하며, Pitch 는 음정, 진동수를 의미하고, Timbre 는 음색, 소리 , 감각 등을 나타낸다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-23-04.png)
<br></br>

위 그림에서 첫 번째 줄은 주파수를 타나낸 것이다. 주파수에 따라 소리의 높이가 달라짐을 볼 수 있다.

두 번째 그림은 진폭에 따른 소리의 강도 (크기) 를 나타내며, 이 두 줄의 그래프는 주파수 혹은 진폭의 변화만 있을 뿐 그래프의 형태는 동일한 것을 볼 수 있다.

세 번째 그림은 그래프 모양이 다른 것을 볼 수 있는데, 음정, 음색, 맵시가 달라지기 때문이다.
<br></br>

### 주파수란?

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-23-05.max-800x600.png)
<br></br>
주파수란 소리의 높낮이를 결정하는 요소이다.

소리는 공기의 압축이며, 주파수는 The Number of compression 으로 표현한다. 즉, 1 초 동안 진동 횟수를 의미하며, Hz 단위를 사용한다. 1 Hz 는 1 초에 한번 진동한다는 것이다.

주기 (Period) 란 파동이 한 번 진동하는데 걸리는 시간이나 길이를 의미하며, sin 함수의 주기는 $(\frac{2\pi}{\omega}, \ \omega = 주파수)$ 로 나타난다. 
<br></br>

### 복합파 (Complex Wave) 란?

복합파란 서로 다른 주파수를 가진 소리들이 뒤엉켜 만들어지며, 일반적으로 사용하는 소리는 복합파이다.

복수의 서로 다른 정현파 (Sine Wave) 의 합으로 이루어지는 파형이며, 다음과 같은 수식으로 표현이 가능하다.

$$ x(n)\approx \sum_{k=0}^{K}a_k(n)cos(\phi _k(n))+e(n) $$
$$\begin{align}
where, a_k & =\text{instantaneous  amplitude} \\
\phi_k & =\text{instantaneous phase} \\
e(n) & = \text{residual (noise)} \\
\end{align}$$
<br></br>

### 오디오 데이터의 디지털화

컴퓨터는 0 과 1 로만 이루어진 데이터만 인식할 수 있다. 오디오 데이터도 예외는 아니다.

이렇게 컴퓨터가 오디오 데이터를 이애하는 과정을 알아보자.

연속적인 아날로그 신호 중 가장 단순한 형태인 사인 함수의 수식은 다음과 같다.

$$ Asin(2\pi ft - \phi) $$

$A$ 는 진폭으로 위, 아래로 움직이는 소리의 크기를 나타낸다. $f$ 는 주파수로 초당 진동 횟수, 소리의 세기를 나타내며, $\phi$ 는 위상, $t$ 는 시간을 나타낸다.
<br></br>
> 아날로그 신호의 표본화 (Sampling)
```python
import numpy as np
import matplotlib.pyplot as plt

def single_tone(frequecy, sampling_rate=16000, duration=1):
    t = np.linspace(0, duration, int(sampling_rate))
    y = np.sin(2 * np.pi * frequecy * t)
    return y

y = single_tone(400)
```
표본화 과정에서는 시간축 방향에서 일정 간격으로 샘플을 추출하여 이산 신호 (discrete signal) 로 변환시킨다.

sampling rate 를 16,000 으로 지정하였고, duration 은 1 초, 주파수는 400 인 사인함수를 정의하였다.

sampling rate 란 초당 샘플링 횟수를 의미하며, 샘플링이란 1 초의 연속적인 신호를 몇 개의 숫자로 나눌지 표현하는 것을 의미한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-23-07.max-800x600.png)
<br></br>
sampling rate 는 어떤 신호의 최대 주파수의 2배 이상의 속도로 균일한 간격의 표본화를 실시하는 나이키스트-섀넌 표본화에 따라 Sampling rate가 결정된다. 

일반적으로 사용되는 주파수 영역대는 16kHz, 44.1kHz를 많이 사용하며 16 kHz 는 보통 말할 때 많이 사용되는 주파수 영역대이고, 44.1 kHz 는 음악에서 많이 사용하는 주파수영역대이다. 

따라서 음성 데이터를 사용하기에 sampling rate 를 16,000 으로 지정한 것이며, 단위는 초 단위 이다.

+ 참고 : [나이키스트-섀넌 표본화](https://linecard.tistory.com/20)
<br></br>
> 아날로그 신호인 사인함수 시각화
```python
plt.plot(y[:41])
plt.show()
```
1 사이클 동안 나타나는 이산 시간 연속크기 신호를 출력한다.
<br></br>

### 표본화 (Sampling), 양자화 (Quantization), 부호화(Encoding)

연속적인 아날로그 신호를 컴퓨터 입력으로 사용할 경우 표본화 (Sampling), 양자화 (Quantization), 부호화 (Encoding) 을 거치게되며, 이진 디지털 신호 (Binary Digital Signal) 로 변환되어 인식한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-23-06_wVIMJrD.png)
<br></br>
위 그림은 오디오 신호를 표본화, 양자화 하는 과정을 나타낸 그림이다.

아날로그 데이터에서 일정 간격으로 표본을 채취하는 방식으로 이산적 데이터를 구한다. 

하지만 디지털화를 위해서 이산적 데이터의 값 자체가 소수점 아래로 무한히 정밀해지며, 일정 간격으로 값을 근사하여 구할 수 밖에 없다. 이 과정을 양자화 라고 한다.

양자화 과정에서 원본 아날로그 데이터와의 차이가 발생하고 왜곡이 발생한다.

Sampling Rate 가 클수록 기대 왜곡치가 적어지며, 원본에 가까운 형태로 변환된다.

이렇게 표본화, 양자화 과정을 거쳐 마지막 부호화 과정을 통해 원본 아날로그 수치가 최종적인 디지털 표현 (0, 1 로만 이루어진 오디오 데이터) 을 얻게 된다. 
<br></br>

### Wave Data 분석

음성 인식 모델을 위한 학습 데이터를 분석해보자.

먼저 모델 구성에 필요한 데이터를 다운받자.
<br></br>
> 데이터 다운로드
```bash
$ mkdir -p ~/aiffel/speech_recognition/data $ mkdir -p ~/aiffel/speech_recognition/models 
$ wget https://aiffelstaticdev.blob.core.windows.net/dataset/speech_wav_8000.npz -P ~/aiffel/speech_recognition/data
```
<br></br>
#### Wave 데이터 형식 뜯어보기

-   Audio 데이터는 이미지 데이터보다 낮은 차원의 데이터를 다룬다. 1 개의 wav 파일은 1 차원으로 이루어진 시계열 데이터이며, 실제로는 여러 소리 파형이 합쳐진 복합파로 봐야한다.

#### 간단한 단어 인식을 위한 훈련데이터셋

-   짧은 단어의 라벨이 달려 있어, 음성들을 텍스트로 구분하는 모델로도 학습이 가능하다.

#### Bits per sample

-   **샘플 하나마다 소리의 세기를 몇 비트로 저장했는지를 나타낸다.**
-   값이 커질 수록 세기를 정확하게 저장할 수 있다.  
    예를 들어, Bits rate가 16 bits 라면, 소리의 세기를  216, 즉 65,536 단계로 표현할 수 있다.
-   4 bits / 8 bits unsigned int / 16 bits int / 24 bits / 32 bits float 등의 자료형으로 표현된다.

#### Sampling frequency

-   샘플링 주파수라는 단어이며, 소리로부터 초당 샘플링한 횟수를 의미한다.
-   샘플링은 원래 신호를 다시 복원할 수 있는 나이퀴스트 (Nyquist) 샘플링 룰에 따라서, 복원해야 할 신호 주파수의 2배 이상으로 샘플링 해야한다.
-   가청 주파수 20 ~ 24 kHz 를 복원하기 위해 사용하며, 음원에서 많이 쓰이는 값은 44.1 kHz 이다.

#### Channel

-   각 채널별로 샘플링된 데이터가 따로 저장되어 있습니다.
-   2채널(Stereo) 음원을 재생하면 왼쪽(L)과 오른쪽(R) 스피커에 다른 값이 출력됩니다.
-   1채널(Mono) 음원의 경우 왼쪽(L) 데이터만 있으며, 재생시엔 왼쪽(L)과 오른쪽(R) 스피커에 같은 값이 출력됩니다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-23.max-800x600.png)

-   1000 Hz 신호를 저장한 wav 파일이다.
-   Sample rate 는 48000 Hz, 즉 1 초 재생하는데 필요한 Sample 수는 48000 개이다.
-   모든 샘플은 -1 ~ 1 사이의 정해진 bits per sample 의 값으로 표현됩니다.
<br></br>
> 데이터 결로 설정
```python
import numpy as np
import os

data_path = os.getenv("HOME")+'/speech_recognition/data/speech_wav_8000.npz'
speech_data = np.load(data_path)
```
<br></br>
### 데이터셋 살펴보기
<br></br>
> 데이터셋 샘플 확인하기
```python
print("Wave data shape : ", speech_data["wav_vals"].shape)

print("Label data shape : ", speech_data["label_vals"].shape)
```
- npz 파일로 이뤄진 데이터이며, 각각 데이터는 "wav_vals", "label_vals" 로 저장되어있다.
-   데이터셋은 1 초 길이의 오디오 음성데이터 50620 개로 이뤄져 있다.
-   주어진 데이터의 원래 Sample rate 는 16,000 이지만, 8,000 으로 re-sampling 해 사용한다.
-   모두 1 초의 길이를 가지는 오디오 음성데이터이여서 각각 8,000 개의 sample data 를 가지고 있다.
<br></br>
> 데이터 확인
```python
import IPython.display as ipd
import random

# 데이터 선택 (랜덤하게 선택하고 있으니, 여러번 실행해 보세요)
rand = random.randint(0, len(speech_data["wav_vals"]))
print("rand num : ", rand)

sr = 8000 # 1초동안 재생되는 샘플의 갯수
data = speech_data["wav_vals"][rand]
print("Wave data shape : ", data.shape)
print("label : ", speech_data["label_vals"][rand])

ipd.Audio(data, rate=sr)
```
<br></br>

## Train / Test 데이터셋 구성하기

### Label Data 처리

현재 음성 단어의 정답은 Text 형태로 구성되어 있다. 학습을 위해서는 Text 데이터를 학습 가능한 형태로 만들어 줘야 한다.

현재 Label 은 `['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go' ]` 로 이루어져있으며, 이 외의 데이터는 `['unknow', 'slience']` 로 분류되어 있다.
<br></br>
> Text 로 이루어진 라벨 데이터를 index 형태로 변환
```python
target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

label_value = target_list
label_value.append('unknown')
label_value.append('silence')

print('LABEL : ', label_value)

new_label_value = dict()
for i, l in enumerate(label_value):
    new_label_value[l] = i
label_value = new_label_value

print('Indexed LABEL : ', new_label_value)
```
int 형태로 바꿔주는 index 작업으로 label data 를 더 쉽게 사용할 수 있다.
<br></br>
> 모든 label data 를 index 형태로 변환
```python
temp = []
for v in speech_data["label_vals"]:
    temp.append(label_value[v[0]])
label_data = np.array(temp)

label_data
```
<br></br>

### 학습을 위한 데이터 분리

> sklearn 의 train_test_split 함수를 이용한 데이터 분리
```python
from sklearn.model_selection import train_test_split

sr = 8000
train_wav, test_wav, train_label, test_label = train_test_split(speech_data["wav_vals"], 
                                                                label_data, 
                                                                test_size=0.1,
                                                                shuffle=True)
print(train_wav)

train_wav = train_wav.reshape([-1, sr, 1]) # add channel for CNN
test_wav = test_wav.reshape([-1, sr, 1])
```
<br></br>
> 나눠진 데이터셋 확인
```python
print("train data : ", train_wav.shape)
print("train labels : ", train_label.shape)
print("test data : ", test_wav.shape)
print("test labels : ", test_label.shape)
```
<br></br>

### Hyper - Parameters Setting

> 학습을 위한 하이퍼 파라미터  설정
```python
batch_size = 32
max_epochs = 10

# the save point
checkpoint_dir = os.getenv('HOME')+'/aiffel/speech_recognition/models/wav'

checkpoint_dir
```
하이퍼 파라미터와, 모델 체크포인트 Callback 함수를 설정하거나, 모델을 불러올때 사용하는 체크 포인트 저장을 위한 경로 설정.
<br></br>

### Data Setting

`tf.data.Dataset`을 이용해서 데이터셋을 구성해야 한다. Tensorflow 에 포함된 이 데이터셋 관리 패키지는 데이터셋 전처리, 배치처리 등을 쉽게 할 수 있도록 해 준다.

`tf.data.Dataset.from_tensor_slices` 함수에 return 받길 원하는 데이터를 튜플 (data, label) 형태로 넣어서 사용할 수 있다.

`map` 함수는 dataset 이 데이터를 불러올때마다 동작시킬 데이터 전처리 함수를 매핑해 주는 역할을 한다.

첫번째 `map` 함수는 `from_tensor_slice` 에 입력한 튜플 형태로 데이터를 받으며 return 값으로 어떤 데이터를 반환할지 결정하며, `map` 함수는 중첩해서 사용이 가능하다.
<br></br>
> `map` 함수에 넘겨줄 데이터 전처리를 위한 함수 생성
```python
def one_hot_label(wav, label):
    label = tf.one_hot(label, depth=12)
    return wav, label
```
<br></br>
> `tf.data.Dataset` 함수 생성
```python
import tensorflow as tf

# for train
train_dataset = tf.data.Dataset.from_tensor_slices((train_wav, train_label))
train_dataset = train_dataset.map(one_hot_label)
train_dataset = train_dataset.repeat().batch(batch_size=batch_size)
print(train_dataset)

# for test
test_dataset = tf.data.Dataset.from_tensor_slices((test_wav, test_label))
test_dataset = test_dataset.map(one_hot_label)
test_dataset = test_dataset.batch(batch_size=batch_size)
print(test_dataset)
```
batch 는 dataset 에서 제공하는 튜플 형태의 데이터를 얼마나 가져올지 결정하는 함수이다.
<br></br>

## Wave Classification 모델 구현

### Model

음성 데이터는 1 차원 데이터로, 1 차원 데이터 형식에 맞게 모델을 구성해줘야한다.

모델 구성에 `Conv1D` layer를 이용하며, Conv, batch norm, dropout, dense layer 등을 이용해 모델을 구성한다.
<br></br>
> 모델 구성
```python
from tensorflow.keras import layers

input_tensor = layers.Input(shape=(sr, 1))

x = layers.Conv1D(32, 9, padding='same', activation='relu')(input_tensor)
x = layers.Conv1D(32, 9, padding='same', activation='relu')(x)
x = layers.MaxPool1D()(x)

x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
x = layers.MaxPool1D()(x)

x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
x = layers.MaxPool1D()(x)

x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
x = layers.MaxPool1D()(x)
x = layers.Dropout(0.3)(x)

x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

output_tensor = layers.Dense(12)(x)

model_wav = tf.keras.Model(input_tensor, output_tensor)

model_wav.summary()
```
<br></br>

### Loss

현재 12 개의 라벨을 가지고 있으며, 해당 라벨에 해당하는 클래스를 구분하기 위해서 Multi - class classification 이 필요하다. 이를 수행하기 위해 Loss 로 Categorical Cross - Entropy Loss 를 사용할 수 있다.

<br></br>
> Loss 설정
```python
optimizer=tf.keras.optimizers.Adam(1e-4)
model_wav.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             optimizer=optimizer,
             metrics=['accuracy'])
```
<br></br>

### Training

#### Callback


callback 이란 모델을 재사용하기위해서 모델 가중치를 저장하는 함수를 의미한다.

`model.fit`  함수를 이용할 때, callback 함수를 이용해서 학습 중간 중간 원하는 동작을 하도록 설정할 수 있다. 
<br></br>
> Callback 모델 생성
```python
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 mode='auto',
                                                 save_best_only=True,
                                                 verbose=1)
```
`Model Checkpoint callback` 은 모델을 학습을 진행하며, `fit` 함수내 다양한 인자를 지정해 모니터하며 동작하게 설정할 수 있다. 

validation loss 를 모니터하며, loss 가 낮아지면 모델 파라미터를 저장하도록 설정하였다.
<br></br>

> 모델 학습
```python
#30분 내외 소요 (메모리 사용량에 주의해 주세요.)
history_wav = model_wav.fit(train_dataset, epochs=max_epochs,
                    steps_per_epoch=len(train_wav) // batch_size,
                    validation_data=test_dataset,
                    validation_steps=len(test_wav) // batch_size,
                    callbacks=[cp_callback]
```
<br></br>

### 학습 결과 시각화
<br></br>
> 학습 결과 시각화
```python
import matplotlib.pyplot as plt

acc = history_wav.history['accuracy']
val_acc = history_wav.history['val_accuracy']

loss=history_wav.history['loss']
val_loss=history_wav.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
model.fit 함수는 학습 동안의 결과를 return 하며, return 값을 기반으로 loss 와 accuracy 를 그래프로 시각화 하였다.

fit 함수에서 전달받은 loss, accuracy 값을 통해 모델 학습 현황을 확인할 수 있다.

train loss 와 val_loss 의 차이가 커지는 경우 오버피팅이 일어나는 것이기 때문에 이를 수정할 필요가 있다.
<br></br>

### Evaluation

이제 테스트 데이터셋을 통해 모델 성능을 평가 해보자.
<br></br>
> checkpoint callback 함수가 저장한 weight 를 다시 불러와서 테스트 준비
```python
model_wav.load_weights(checkpoint_dir)
```
<br></br>
> 테스트 데이터셋을 이용한 모델 성능 평가
```python
results = model_wav.evaluate(test_dataset)
```
<br></br>
> 모델 성능 시각화
```python
# loss
print("loss value: {:.3f}".format(results[0]))
# accuracy
print("accuracy value: {:.4f}%".format(results[1]*100))
```
<br></br>

### Model Test

> Test data 셋을 골라 직접 들어보고 모델의 예측이 맞는지 확인
```python
inv_label_value = {v: k for k, v in label_value.items()}
batch_index = np.random.choice(len(test_wav), size=1, replace=False)

batch_xs = test_wav[batch_index]
batch_ys = test_label[batch_index]
y_pred_ = model_wav(batch_xs, training=False)

print("label : ", str(inv_label_value[batch_ys[0]]))

ipd.Audio(batch_xs.reshape(8000,), rate=8000)
```
<br></br>
> 테스트셋 라벨과 모델의 실제 예측 결과 비교
```python
if np.argmax(y_pred_) == batch_ys[0]:
    print("y_pred: " + str(inv_label_value[np.argmax(y_pred_)]) + '(Correct!)')
else:
    print("y_pred: " + str(inv_label_value[np.argmax(y_pred_)]) + '(Incorrect!)')
```
<br></br>

## Skip - Connection Model 을 추가해보자.

### Skip - Connection Model 구현

ResNet 등의 이미지 처리 모델을 보면 Skip - Connection 을 활용한 모델이 훨씬 안정적으로 높은 성능을 나타내는 것을 알 수있다.

이러한 Skip - Connection 을 음성 처리 모델에 적용해보자.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/skip-connection.max-800x600.png)
<br></br>
위 그림은 ResNet 에서 사용된 Skip - Connection 의 개념을 나타내고 있다.

위쪽으 데이터가 레이어를 뛰어 넘어 레이어를 통과한 값에 더해주는 형식으로 구현할 수 있으며, Concat 을 이용해 구현할 수 있다.

Concat 사용 양식 : `tf.concat([#layer output tensor, layer output tensor#], axis=#)`
<br></br>
> 1 차원 오디오 데이터를 처리하는 모델 구현
> (Skip - Connection 추가)
```python
input_tensor = layers.Input(shape=(sr, 1))

x = layers.Conv1D(32, 9, padding='same', activation='relu')(input_tensor)
x = layers.Conv1D(32, 9, padding='same', activation='relu')(x)
skip_1 = layers.MaxPool1D()(x)

x = layers.Conv1D(64, 9, padding='same', activation='relu')(skip_1)
x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
x = tf.concat([x, skip_1], -1)
skip_2 = layers.MaxPool1D()(x)

x = layers.Conv1D(128, 9, padding='same', activation='relu')(skip_2)
x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
x = tf.concat([x, skip_2], -1)
skip_3 = layers.MaxPool1D()(x)

x = layers.Conv1D(256, 9, padding='same', activation='relu')(skip_3)
x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
x = tf.concat([x, skip_3], -1)
x = layers.MaxPool1D()(x)
x = layers.Dropout(0.3)(x)

x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

output_tensor = layers.Dense(12)(x)

model_wav_skip = tf.keras.Model(input_tensor, output_tensor)

model_wav_skip.summary()
```
모델 구성 외에 다른 과정은 동일하다.
<br></br>
> 모델 loss 설정
```python
optimizer=tf.keras.optimizers.Adam(1e-4)
model_wav_skip.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             optimizer=optimizer,
             metrics=['accuracy'])
```
<br></br>
> Callback 모델 생성
```python
# the save point
checkpoint_dir = os.getenv('HOME')+'/aiffel/speech_recognition/models/wav_skip'

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 mode='auto',
                                                 save_best_only=True,
                                                 verbose=1)
```
<br></br>
> 모델 학습
```python
#30분 내외 소요
history_wav_skip = model_wav_skip.fit(train_dataset, epochs=max_epochs,
                    steps_per_epoch=len(train_wav) // batch_size,
                    validation_data=test_dataset,
                    validation_steps=len(test_wav) // batch_size,
                    callbacks=[cp_callback]
                    )
```
<br></br>
> 학습 과정 시각화
```python
import matplotlib.pyplot as plt

acc = history_wav_skip.history['accuracy']
val_acc = history_wav_skip.history['val_accuracy']

loss=history_wav_skip.history['loss']
val_loss=history_wav_skip.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
<br></br>
> 테스트 데이터셋을 활용한 모델 성능 평가
```python
# Evaluation 

model_wav_skip.load_weights(checkpoint_dir)
results = model_wav_skip.evaluate(test_dataset)

# loss
print("loss value: {:.3f}".format(results[0]))
# accuracy
print("accuracy value: {:.4f}%".format(results[1]*100))
```
<br></br>
> Test data 셋을 골라 직접 들어보고 모델의 예측이 맞는지 확인
```python
# Test 

inv_label_value = {v: k for k, v in label_value.items()}
batch_index = np.random.choice(len(test_wav), size=1, replace=False)

batch_xs = test_wav[batch_index]
batch_ys = test_label[batch_index]
y_pred_ = model_wav_skip(batch_xs, training=False)

print("label : ", str(inv_label_value[batch_ys[0]]))

ipd.Audio(batch_xs.reshape(8000,), rate=8000)
```
<br></br>
> 테스트셋 라벨과 실제 모델이 예측한 결과를 비교
```python
if np.argmax(y_pred_) == batch_ys[0]:
    print("y_pred: " + str(inv_label_value[np.argmax(y_pred_)]) + '(Correct!)')
else:
    print("y_pred: " + str(inv_label_value[np.argmax(y_pred_)]) + '(Incorrect!)')
```
<br></br>

## Spectrogram

위에서 음성 데이터를 1 차원 시계열 데이터로 해석하고, Wavefrom 해석을 통해 다뤘다.

하지만 많은 음성 데이터들은 복합파이므로, 다양한 파형을 주파수 대역별로 가지고 있다.

### 푸리에 변환 (Fourier Transform)

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-23-08.png)
<br></br>
푸리에 변환은 임의의 입력 신호를 다양한 주파수를 갖는 주기함수 (복수 지수함수)들이 합으로 분해하여 표현하는 것이다.

즉, 복합파가 가진 다양한 파형을 주파수 대역별로 나눠 해석할 수 있게 해준다.

푸리에 변환을 수식으로 표현하면 $$A_k = \frac{1}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(t)exp(-i\cdot 2\pi\frac{k}{T}t)dt$$

과 같다.

### 오일러 공식 (지수함수와 주기함수와의 관계)

$cos2\pi kT, jsin2\pi kT$ 함수는 주기와 주파수를 가지는 주기함수이다.

푸리에 변환은 입력 신호가 무엇인지 상관없이 sin, cos 과 같은 주기함수들의 합으로 항상 분해가 가능하다.

이를 수식으로 나타내면 다음과 같다.

$$e^{i\theta }=cos\theta + isin\theta$$
$$exp(i\cdot 2\pi \frac{k}{T}t)=cos(2\pi \frac{k}{T})+jsin(2\pi \frac{k}{T})$$
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-23-09.max-800x600.png)
<br></br>

푸리에 변환이 끝나면 실수부와 허수부를 가지는 복소수를 얻을 수 있다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-23-10.max-800x600.png)
<br></br>
이렇게 구해진 복소수의 절대값은 주파수의 강도 (Spectrum Magnitude) 라고 하며, 복소수가 가지는 Phase 는 주파수의 위상 (Phase Spectrum) 이라고 한다. 

이를 수식으로 나타내면 다음과 같다.

$$\left | X(k) \right |=\sqrt{X^2_R(k)+X^2_I(k) }$$
$$\angle X = \pi(k)=tan^{-1}\frac{X_I(k)}{X_R(k)}$$
<br></br>

### STFT (short Time Fourier Transform)

FFT 는 시간의 흐름에 따라 신호의 주파수가 변했을 때, 어느 시간대에 주파수가 변하는지 알 수 없다는 단점이 존재한다.

이러한 단점을 개선한 방법이 STFT 이며, STFT 는 시간의 길이를 나눠 푸리에 변환을 실시한다.

STFT 를 수식으로 표현하면 다음과 같다.

$$X(l,k)=\sum_{n=0}^{N-1}\omega (n)x(n+lH)exp^{\frac{-2\pi kn}{N}}$$
<br></br>
-   N 은 FFT size 이고, Window 를 얼마나 많은 주파수 밴드로 나누는 가를 의미.

-   Duration 은 sampling rate 를 window 로 나눈 값이다. $T=window/sampling rate$ 이며, duration 은 신호주기보다 5배이상 길게 잡아야한다. ($T(window)=5T(Signal)$). 예를 들어 440 Hz 신호의 window size 는 5(1/440) 이 됩니다.

-   $ω (n)$는 window 함수를 나타내며, 일반적으로는 hann window가 많이 쓰인다.
 
-   n 는 window size 이다. window 함수에 들어가는 sample 의 양입니다. n이 작을 수록 low-frequency resolution 와 high-time resolution 을 가지게 되고, n이 길수록 high-frequency 를 가지고 되고 low-time resolution을 가지게 된다.

-   H 는 hop size 를 의미한다. 윈도우가 겹치는 사이즈이며, 일반적으로는 1/4 정도를 겹치게 한다.
<br></br>
STFT 의 결과는 시간의 흐름에 따른 주파수 영역별 Amplitude 를 반환한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-23-11.png)
<br></br>

### Spectrogram 이란?

스펙트로그램이란 Wav 데이터를 해석하는 방법 중 하나로 일정 시간동안 Wav 데이터 안의 주파수들이 얼마나 포함되어 있는지를 보여준다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/spectogram.max-800x600.png)
<br></br>
위 그림은 스펙트로그램을 실시한 모습이며, X 축은 시간, Y 축은 주파수를 나타내며, 해당 시간 / 주파수에서의 음파 강도에 따라 밝은색으로 표현된다.

wav 데이터가 단위 시간만큼 Short Time Fourier Transform 을 진행해 매 순간의 주파수 데이터들을 얻어서 Spectrogram 을 완성한다.

<br></br>
> Spectrogram 을 위한 라이브러리 설치 및 Wav 데이터를 Spectrogram 하기 위한 함수 생성
```python
import librosa

def wav2spec(wav, fft_size=258): # spectrogram shape을 맞추기위해서 size 변형
    D = np.abs(librosa.stft(wav, n_fft=fft_size))
    return D
```
<br></br>
> Wav 데이터를 몇 개 뽑아 Spectrogram 수행
```python
# 위에서 뽑았던 sample data
spec = wav2spec(data)
print("Waveform shape : ",data.shape)
print("Spectrogram shape : ",spec.shape)
```
Spectrogram 을 통해 1 차원 Wav 데이터를 2 차원 데이터로 변환하였다.
<br></br>
> 변환된 Spectrogram 출력
```python
import librosa.display

librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.xticks(range(0, 1))
plt.tight_layout()
plt.show()
```
<br></br>
