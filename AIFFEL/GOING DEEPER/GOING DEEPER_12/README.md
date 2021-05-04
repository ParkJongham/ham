# 12. 직접 만들어보는 OCR


### 학습 목표

1.  Text Recognition 모델을 직접 구현해 봅니다.
2.  Text Recognition 모델 학습을 진행해 봅니다.
3.  Text Detection 모델과 연결하여 전체 OCR 시스템을 구현합니다.

<br></br>
## Overall structure of OCR

![](https://aiffelstaticprd.blob.core.windows.net/media/images/e-23-1.ocr-system.max-800x600.png)
<br></br>
OCR 이란 이미지 속 문자를 Bounding Box 로 찾아내고 해당 bounding Box 안에 어떤 텍스트가 포함되어 있는지 알아내는 것이다.

이미지 속에서 문자를 찾아내는 기법을 Text Detection 이라고 하며, Segmentation 기반의 CRAFT 를 활용한 Keras - ocr 을 통해 수행할 수 있다.

문자를 검출해내고 해당 문자가 어떤 텍스트인지 알아내기 위한 기법은 Recognition 이라고 한다.

+ 참고 : 
	-   [keras-ocr 공식 github](https://github.com/faustomorales/keras-ocr)
	-   [CRAFT: Character-Region Awareness For Text detection](https://arxiv.org/pdf/1904.01941.pdf)
	    -   [CRAFT Pytorch 공식 implementation](https://github.com/clovaai/CRAFT-pytorch)
	    -   [CRAFT Keras 버전 github](https://github.com/notAI-tech/keras-craft)
<br></br>
Text Recognittion 의 모델로는 CNN 과 RNN 의 아이디어를 결합하고 CTC 로 학습된 초기 모델 CRNN 이 있으며, 이 RCNN 은 Keras - ocr 에서 제공하는 Recognition 에 활용되고 있다.

최근 2019 년 네이버 Clova 에서 발표한 논문을 통해 Text Recognition 의 발전을 살펴볼 수 있다.

+ 참고 : [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/pdf/1904.01906.pdf)
<br></br>

## Dataset for OCR

OCR 의 데이터셋에 필요한 텍스트 정보는 사람이 직업 입력해야한다는 큰 차이점이 있다. 때문에 OCR 데이터를 대량으로 구축하기 위해서는 많은 자원이 소요된다.

이러한 문제를 해결하기위한 방안 중 하나로 컴퓨터로 대량의 문자 이미지 데이터를 만들어 내는 방법이 있다.

이는 언어, 폰트, 배치, 크기 등을 원하는데로 만들 수 있다는 장점이 있다.

[What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://github.com/clovaai/deep-text-recognition-benchmark) 논문에서는 Recognition 모델의 정량적 평가를 위해 `MJ Synth` 와 `SynthText` 라는 데이터셋을 활용한다.

1. [MJ Synth](http://www.robots.ox.ac.uk/~vgg/data/text/)

2. [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
<br></br>

우리가 만들어 볼 Recognition model 학습을 위해 `MJ Synth` 를 사용할 것이다.


<br></br>
> 데이터셋 다운로드
> [Dropbox-data_lmdb_release](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0)
```bash
$ mkdir -p ~/aiffel/ocr 
$ cd ~/aiffel/ocr $ wget https://www.dropbox.com/sh/i39abvnefllx2si/AABX4yjNn2iLeKZh1OAwJUffa/data_lmdb_release.zip $ unzip data_lmdb_release.zip 
$ mv data_lmdb_release/training/MJ . # data_lmdb_release/training/MJ 아래의 데이터만 ~/aiffel/ocr 아래로 가져옵니다.  # 이후 불필요한 data_lmdb_release.zip 및 data_lmdb_release 하위의 남은 데이터는 삭제하셔도 무방합니다.
```
먼저 데이터를 다운 받자. 데이터는 네이버 Clova 에서 제공하는 데이터셋이며, path 설정 후 아래 링크의 training 폴더에서 `data_lmdb_release.zip` 파일을 다운로드하자.
<br></br>
> 다운받은 데이터 경로 설정
```python
import os

path = os.path.join(os.getenv('HOME'),'aiffel/ocr')
os.chdir(path)

print(path)
```
<br></br>

## Recognition Model (01)

CRNN 구조를 활용하여 Text Recognition 모델 설계를 해보자.

+ 참고 : [논문 : An End-to-End Trainable Neural Network for Image-based SequenceRecognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/e-23-2.crnn.png)<br></br>

위 그림은 CRNN 의 구조를 나타낸다. 입력 이미지는 Convolution Layer 를 통해 특성을 추출한다.

Recurrent Layer 는 Convolution Layer 에서 추출된 특성의 전체적인 Context 를 파악하며, 다양한 Ouput 의 크기에 대응이 가능하다.

Transcription Layer (Fully Connected Layer) 는 Step 마다 어떤 문자의 확률이 높은지 예측한다.

<br></br>
+ RCNN 의 구조 상세 : 

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/e-23-3.crnn_structure.png)
<br></br>
> 모델 구현을 위해 몇 개의 class 가 필요한기 확인
```python
NUMBERS = "0123456789"
ENG_CHAR_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGET_CHARACTERS = ENG_CHAR_UPPER + NUMBERS
print(f"The total number of characters is {len(TARGET_CHARACTERS)}")
```
36 개의 클래스가 필요함을 알 수 있다.
<br></br>

> lmdb 라이브러리 설치 및 사용할 라이브러리 임포트, MJ 데이터셋의 위치 확인 및 이동
> (LMDB는 Symas에서 만든 Lightning Memory-Mapped Database)
```python
pip install lmdb

import re
import six
import math
import lmdb
import os
import numpy as np
import tensorflow as tf

from PIL import Image

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

BATCH_SIZE = 128
HOME_DIR = os.getenv('HOME')+'/aiffel/ocr/data_lmdb_release'

# 로컬 사용자
TRAIN_DATA_PATH = HOME_DIR+'/training/MJ/MJ_train'
VALID_DATA_PATH = HOME_DIR+'/training/MJ/MJ_valid'
TEST_DATA_PATH = HOME_DIR+'/training/MJ/MJ_test'

# 클라우드 사용자는 아래 주석을 사용해 주세요.
# TRAIN_DATA_PATH = HOME_DIR+'/data/MJ/MJ_train'
# VALID_DATA_PATH = HOME_DIR+'/data/MJ/MJ_valid'
# TEST_DATA_PATH = HOME_DIR+'/data/MJ/MJ_test'
print(TRAIN_DATA_PATH)
```
<br></br>

## Recognition Model (02). Input Image

> 데이터셋의 실제 이미지 및 라벨 확인
```python
from IPython.display import display

env = lmdb.open(TRAIN_DATA_PATH, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    for index in range(1, 5):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key).decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')

        except IOError:
            img = Image.new('RGB', (100, 32))
            label = '-'
        width, height = img.size
        print('original image width:{}, height:{}'.format(width, height))
        
        target_width = min(int(width*32/height), 100)
        target_img_size = (target_width,32 )
        
        print('target_img_size:{}'.format(target_img_size))
        
        img = np.array(img.resize(target_img_size)).transpose(1,0,2)
       
        print('display img shape:{}'.format(img.shape))
        print('label:{}'.format(label))
        display(Image.fromarray(img.transpose(1,0,2).astype(np.uint8)))
```
대부분의 이미지는 height 31, 최대 height 32 로 되어 있으며, width 는 문자열 길이에 따라 다른 것으로 나타난다.
<br></br>
> lmdb 를 활용, 케라드 모델 학습용 MJ Synth 데이터 셋 클래스 구현
```python
class MJDatasetSequence(Sequence):
    def __init__(self, 
                      dataset_path,
                      label_converter,
                      batch_size=1,
                      img_size=(100,32),
                      max_text_len=22,
                      is_train=False,
                      character=''
                ):
        
        self.label_converter = label_converter
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.character = character
        self.is_train = is_train
        self.divide_length = 100

        self.env = lmdb.open(dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            self.num_samples = int(num_samples)
            self.index_list = [index + 1 for index in range(self.num_samples)]
        
    def __len__(self):
        if self.is_train:
            return math.ceil(self.num_samples/self.batch_size/self.divide_length)
        return math.ceil(self.num_samples/self.batch_size/self.divide_length)
    
    # index에 해당하는 image와 label을 가져오는 메소드
    def _get_img_label(self, index):
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')

            except IOError:
                img = Image.new('RGB', self.img_size)
                label = '-'
            width, height = img.size
            
            target_width = min(int(width*self.img_size[1]/height), self.img_size[0])
            target_img_size = (target_width,self.img_size[1] )
            img = np.array(img.resize(target_img_size)).transpose(1,0,2)
            label = label.upper()[:self.max_text_len]
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)
    
    # idx번째 배치를 가져오는 메소드
    def __getitem__(self, idx):
        batch_indicies = self.index_list[
            idx*self.batch_size:
            (idx+1)*self.batch_size
        ]
        input_images = np.zeros([self.batch_size, *self.img_size, 3])
        labels = np.zeros([self.batch_size, self.max_text_len], dtype='int64')

        input_length = np.ones([self.batch_size], dtype='int64')*self.max_text_len
        label_length = np.ones([self.batch_size], dtype='int64')

        for i, index in enumerate(batch_indicies):
            img, label = self._get_img_label(index)
            encoded_label = self.label_converter.encode(label)
            width = img.shape[0]
            input_images[i,:width,:,:] = img
            if len(encoded_label) > self.max_text_len:
                continue
            labels[i,0:len(encoded_label)] = encoded_label
            label_length[i] = len(encoded_label)

        inputs = {
            'input_image': input_images,
            'label': labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros([self.batch_size, 1])}
        return inputs, outputs
        
    def on_epoch_end(self):
        self.index_list =  [index + 1 for index in range(self.num_samples)]
        if self.is_train :
            np.random.shuffle(self.index_list)
            return self.index_list
```
dataset_path 는 읽어들일 데이터셋의 경로이며, label_converter 는 문자를 미리정의된 index 로 변환해주는 converter 로 직접 구현하도록 하며, batch_size와 입력이미지 크기 그리고 필터링을 위한 최대 글자 수, 학습대상으로 한정하기 위한 character등을 입력으로 받도록 구현되어 있다.

이미지 데이터를 img, label 의 쌍으로 가져오는 부분은 `_get_img_label()` 메소드에 반영되어있으며, `model.fit()` 에서 호출되는 `__getitem__()` 메소드에서 배치 단위만큼 `_get_img_label()` 를 통해 가져온 데이터셋을 리턴하게 된다.

`_get_img_label()` 를 보면 다양한 사이즈의 이미지를 모두 height 는 32로 맞추고, width 는 최대 100까지로 맞추게끔 가공한다.
<br></br>

## Recognition Model (03). Encode

앞서 출력한 라벨을 보면 평문 텍스트로 구성되어 있었다. 

하지만 이는 모델 학습에 적합한 형태가 아니므로 각 문자를 클래스로 가정하고 Step 에 따른 클랙스 인덱스로 변환해 Encode 를 해줘야한다.

`Label Converter` 클래스가 해당 역할을 해주도록 구현해보자.

`Label Converter` 클래스는 다음과 같은 요소로 구현해야한다.

-   `__init__()`  에서는 입력으로 받은 text 를  `self.dict` 에 각 character 들이 어떤 index 에 매핑되는지 저장한다. 이 character 와 index 정보를 통해 모델이 학습할 수 있는 output 이 만들어지며, 만약  `character='ABCD'` 라면  `'A'` 의 label은 1,  `'B'` 의 label은 2 가 됩니다.
-   공백(blank) 문자를 지정한다. 여기서는 공백 문자를 뜻하기 위해  `'-'` 를 활용하며, label 은 0 으로 지정한다.
-   `decode()` 는 각 index 를 다시 character 로 변환한 후 이어주어 우리가 읽을 수 있는 text 로 바꾸어주는 역할을 한다.
<br></br>

> `Label Converter` 클래스 구현
```python
class LabelConverter(object):
     """ Convert between text-label and text-index """

     def __init__(self, character):
         self.character = "-" + character
         self.label_map = dict()
         for i, char in enumerate(self.character):
             self.label_map[char] = i

     def encode(self, text):
         encoded_label = []
         for i, char in enumerate(text):
             if i > 0 and char == text[i - 1]:
                 encoded_label.append(0)    # 같은 문자 사이에 공백 문자 label을 삽입
             encoded_label.append(self.label_map[char])
         return np.array(encoded_label)

     def decode(self, encoded_label):
         target_characters = list(self.character)
         decoded_label = ""
         for encode in encoded_label:
             decoded_label += self.character[encode]
         return decoded_label
```
<br></br>
> `Label Converter` 클래스가 잘 작동하는지 확인
```python
label_converter = LabelConverter(TARGET_CHARACTERS)

encdoded_text = label_converter.encode('HELLO')
print("Encdoded_text: ", encdoded_text)
decoded_text = label_converter.decode(encdoded_text)
print("Decoded_text: ", decoded_text)
```
연속되는문자가 있을 때 연속되는 문자 사이에 공백이 포함됨을 알 수 있다.
<br></br>

## Recognition Model (04). Build CRNN Model

이제 입력과 출력을 준비했으므로 모델을 구현해보자.

> Keras 에서 제공하는 `K.ctc_batch_cost()`를 활용해서 loss를 계산하도록 `ctc_lambda_func` 생성
```python
def ctc_lambda_func(args): # CTC loss를 계산하기 위한 Lambda 함수
    labels, y_pred, label_length, input_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
```
<br></br>


![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/ctc.png)
<br></br>
위에서 사용한 `K.ctc_batch_cost()` 는 CTC Loss 함수이며, 위 그림과 같은 상황을 다룬다.

입력 길이 T 와 라벨 길이 U 의 단위가 일치하지 않을 때이다. 그림과 같이 라벨은 APPLE 이지만 실제로는 AAAPPPPLLLLEE 처럼 나올 수 있다.

위와 같은 상황이 이미지에서 텍스트 라벨을 추론하는 Text Recognition 에도 동일하게 적용된다.

즉, APPLE 일 때 AAPPPPLLLLEE 를 출력했다면, 최종 추론 결과가 APLE 인지 APPLE 인지 헷갈리게된다.

일반적으로는 APLE 로 추론하게 되며, 이를 방지하기위해 동일 문자가 있을 때는 AP - PLE 로 보정해줄 필요가 있다.

이를 위해 `LabelConverter.encode()` 메소드에 공백 문자 처리 로직을 포함한 것이다.

+ 참고 : [Tensorflow Tutorial - ctc_batch_cost](https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/backend/ctc_batch_cost)
<br></br>

`K.ctc_batch_cost(y_true, y_pred, input_length, label_length)` 에는 4가지 인자가 존재하며, 각 각 

-   `y_true` : tensor (samples, max_string_length) containing the truth labels.
-   `y_pred` : tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.
-   `input_length tensor` : (samples, 1) containing the sequence length for each batch item in y_pred.
-   `label_length tensor` : (samples, 1) containing the sequence length for each batch item in y_true.

를 의미한다. samples 는 배치 사이즈를 의미한다.
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-6-P-example.png)

위 그림은 앞서 살펴본 실제 데이터셋이다.

해당 그림에서 `K.ctc_batch_cost(y_true, y_pred, input_length, label_length)` 에는 4가지 인자는 

-   `y_true` : 실제 라벨  `LUBE`. 그러나 텍스트 라벨 그대로가 아니라, 각 글자를 One-hot 인코딩한 형태로서, max_string_length 값은 모델에서 22 로 지정할 예정
-   `y_pred` : 우리가 만들 RCNN  `모델의 출력 결과`. 그러나 길이는 4 가 아니라 우리가 만들 RNN 의 최종 출력 길이로서 24 가 될 예정
-   `input_length tensor` : 모델 입력 길이 T 로서, 이 경우에는 텍스트의 width 인  `74`
-   `label_length tensor` : 라벨의 실제 정답 길이 U 로서, 이 경우에는  `4` 가 된다.
<br></br>

> `build_crnn_model()` 함수를  구현
> (`K.ctc_batch_cost()`를 활용하여, `image_input`을 입력으로 그리고 마지막 Label을 'output'이라는 이름으로 출력하는 레이어를 갖도록 모델만드는 함수)
```python
def build_crnn_model(input_shape=(100,32,3), characters=TARGET_CHARACTERS):
    num_chars = len(characters)+2
    image_input = layers.Input(shape=input_shape, dtype='float32', name='input_image')

    # Build CRNN model
    conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(image_input)
    conv = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    conv = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    conv = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = layers.MaxPooling2D(pool_size=(1, 2))(conv)
    conv = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.MaxPooling2D(pool_size=(1, 2))(conv)     
    feature = layers.Conv2D(512, (2, 2), activation='relu', kernel_initializer='he_normal')(conv)
    sequnce = layers.Reshape(target_shape=(24, 512))(feature)
    sequnce = layers.Dense(64, activation='relu')(sequnce)
    sequnce = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(sequnce)
    sequnce = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(sequnce)
    y_pred = layers.Dense(num_chars, activation='softmax', name='output')(sequnce)

    labels = layers.Input(shape=[22], dtype='int64', name='label')
    input_length = layers.Input(shape=[1], dtype='int64', name='input_length')
    label_length = layers.Input(shape=[1], dtype='int64', name='label_length')
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [labels, y_pred, label_length, input_length]
    )
    model_input = [image_input, labels, input_length, label_length]
    model = Model(
        inputs=model_input,
        outputs=loss_out
    )
    y_func = tf.keras.backend.function(image_input, [y_pred])
    return model, y_func
```
<br></br>

## Recognition Model (05). Train & Inference

> `MJDatasetSequence` 로 데이터를 분리 및 학습
```python
train_set = MJDatasetSequence(TRAIN_DATA_PATH, label_converter, batch_size=BATCH_SIZE, character=TARGET_CHARACTERS, is_train=True)
val_set = MJDatasetSequence(VALID_DATA_PATH, label_converter, batch_size=BATCH_SIZE, character=TARGET_CHARACTERS)

checkpoint_path = HOME_DIR + '/model_checkpoint.hdf5'

model, y_func = build_crnn_model()
sgd = tf.keras.optimizers.Adadelta(lr=0.1, clipnorm=5)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
ckp = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss',
    verbose=1, save_best_only=True, save_weights_only=True
)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'
)
model.fit(train_set,
        steps_per_epoch=len(val_set),
        epochs=100,
        validation_data=val_set,
        validation_steps=len(val_set),
        callbacks=[ckp])
```
<br></br>
> 테스트셋에서 inference 후 시각화를 통한 확인
```python
from IPython.display import display

test_set = MJDatasetSequence(TEST_DATA_PATH, label_converter, batch_size=BATCH_SIZE, character=TARGET_CHARACTERS)

model, y_func = build_crnn_model()

model.load_weights(checkpoint_path)
input_data = model.get_layer('input_image').output
y_pred = model.get_layer('output').output
model_pred = Model(inputs=input_data, outputs=y_pred)

def decode_predict_ctc(out, chars = TARGET_CHARACTERS, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        indexes = K.get_value(
            K.ctc_decode(
                out, input_length = np.ones(out.shape[0]) * out.shape[1],
                greedy =False , beam_width = beam_width, top_paths = top_paths
            )[0][i]
        )[0]
        text = ""
        for index in indexes:
            text += chars[index]
        results.append(text)
    return results

def check_inference(model, dataset, index = 5):
    for i in range(index):
        inputs, outputs = dataset[i]
        img = dataset[i][0]['input_image'][0:1,:,:,:]
        output = model_pred.predict(img)
        result = decode_predict_ctc(output, chars="-"+TARGET_CHARACTERS)[0].replace('-','')
        print("Result: \t", result)
        display(Image.fromarray(img[0].transpose(1,0,2).astype(np.uint8)))

check_inference(model, test_set, index=10)
```
<br></br>

