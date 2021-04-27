# 08. 도로 영역을 찾자! - 세그멘테이션 모델 만들기

Semantic Segmentation 을 이용한 자율주행 차량이 주행해야 할 도로 영역을 찾는 상황을 가정하고, 모델을 만들어보자.

U - Net 을 사용하여 이미지가 입력되면 도로 영역을 Segmentation 하는 모델을 만들어보자.

최종적으로 만들어 볼 모델은 다음과 같다. 입력 이미지 위에 도로 영역으로 인식한 영역을 오버레이 한 이미지이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/result.gif)

## 실습목표

1.  시맨틱 세그멘테이션 데이터셋을 전처리할 수 있습니다.
2.  시맨틱 세그멘테이션 모델을 만들고 학습할 수 있습니다.
3.  시맨틱 세그멘테이션 모델의 결과를 시각화할 수 있습니다.


## 학습내용

1.  시맨틱 세그멘테이션 데이터셋
2.  시맨틱 세그멘테이션 모델
3.  시맨틱 세그멘테이션 모델 시각화

<br></br>
## 시맨틱 세그멘테이션 데이터셋


### 이미지, 데이터 가져오기

시맨틱 세그멘테이션 으로 도로 영역을 분리하기 위해서 도로의 영역을 라벨로 가진 데이터셋으로 학습 할 수 있도록 파싱해야 한다.

사용할 데이터셋은 KITTI 의 세그멘테이션 데이터를 사용한다.

* 데이터 출처 : http://www.cvlibs.net/datasets/kitti/eval_semantics.php

<br></br>
> 데이터 다운로드 및 작업 디렉토리 설정
```python
$ mkdir -p ~/aiffel/semantic_segmentation/data 
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip 
$ unzip data_semantics.zip -d ~/aiffel/semantic_segmentation/data
```
<br></br>

시맨틱 세그멘테이션으로 찾아야 할 도로는 7 이라는 label 값을 가진다.

* 참고 : [development kit zip 파일의 라벨 정보](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)

<br></br>
## 데이터 로더 (Data loader) 만들기

모델을 학습시킬 수 있는 데이터 로더는 입력값 (224, 224), 출력값 (224, 224) 크기를 갖는 모델을 학습할 수 있도록 데이터셋을 파싱해야 한다.

`albumentations` 을 통해 데이터 로더에 augmentation 을 적용하여 보다 잘 학습할 수 있도록하자. `imguag` 등 다른 augmentation 라이브러리를 사용해도 된다.

또한 학습 데이터의 일정량은 검증 데이터셋으로 활용할 수 있도록 해야한다.

* 참고 : 
	* [Keras Sequence에 기반한 Dataloader](https://hwiyong.tistory.com/241)
	* [Albumentation을 적용한 Keras sequence](https://medium.com/the-artificial-impostor/custom-image-augmentation-with-keras-70595b01aeac)

<br></br>
> 사용 라이브러리 가져오기
```python
pip install albumentations

import os
import math
import numpy as np
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from glob import glob

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
```
<br></br>

> Albumentation 에서 `Compose()` 메서드의 활용을 위한 build_augmentation 함수 생성
> (imgaug 의 `Somtimes()` 와 유사한 기능)
```python
from albumentations import  HorizontalFlip, RandomSizedCrop, Compose, OneOf, Resize

def build_augmentation(is_train=True):
  if is_train:    # 훈련용 데이터일 경우
    return Compose([
                    HorizontalFlip(p=0.5),    # 50%의 확률로 좌우대칭
                    RandomSizedCrop(         # 50%의 확률로 RandomSizedCrop
                        min_max_height=(300, 370),
                        w2h_ratio=370/1242,
                        height=224,
                        width=224,
                        p=0.5
                        ),
                    Resize(              # 입력이미지를 224X224로 resize
                        width=224,
                        height=224
                        )
                    ])
  return Compose([      # 테스트용 데이터일 경우에는 224X224로 resize만 수행합니다. 
                Resize(
                    width=224,
                    height=224
                    )
                ])
```
<br></br>
> 훈련 데이터셋에 augmentation 적용
```python
import os

dir_path = os.getenv('HOME')+'/aiffel/semantic_segmentation/data/training'

augmentation = build_augmentation()
input_images = glob(os.path.join(dir_path, "image_2", "*.png"))

# 훈련 데이터셋에서 5개만 가져와 augmentation을 적용해 봅시다.  
plt.figure(figsize=(12, 20))
for i in range(5):
    image = imread(input_images[i]) 
    image_data = {"image":image}
    resized = augmentation(**image_data, is_train=False)
    processed = augmentation(**image_data)
    plt.subplot(5, 2, 2*i+1)
    plt.imshow(resized["image"])  # 왼쪽이 원본이미지
    plt.subplot(5, 2, 2*i+2)
    plt.imshow(processed["image"])  # 오른쪽이 augment된 이미지
  
plt.show()
```
<br></br>
> 데이터 셋 구성
> (`tf.keras.utils.Sequence` 를 상속받은 generator 형태로 구성)
```python
class KittiGenerator(tf.keras.utils.Sequence):
  '''
  KittiGenerator는 tf.keras.utils.Sequence를 상속받습니다.
  우리가 KittiDataset을 원하는 방식으로 preprocess하기 위해서 Sequnce를 커스텀해 사용합니다.
  '''
  def __init__(self, 
               dir_path,
               batch_size=16,
               img_size=(224, 224, 3),
               output_size=(224, 224),
               is_train=True,
               augmentation=None):
    '''
    dir_path: dataset의 directory path입니다.
    batch_size: batch_size입니다.
    img_size: preprocess에 사용할 입력이미지의 크기입니다.
    output_size: ground_truth를 만들어주기 위한 크기입니다.
    is_train: 이 Generator가 학습용인지 테스트용인지 구분합니다.
    augmentation: 적용하길 원하는 augmentation 함수를 인자로 받습니다.
    '''
    self.dir_path = dir_path
    self.batch_size = batch_size
    self.is_train = is_train
    self.dir_path = dir_path
    self.augmentation = augmentation
    self.img_size = img_size
    self.output_size = output_size

    # load_dataset()을 통해서 kitti dataset의 directory path에서 라벨과 이미지를 확인합니다.
    self.data = self.load_dataset()

  def load_dataset(self):
    # kitti dataset에서 필요한 정보(이미지 경로 및 라벨)를 directory에서 확인하고 로드하는 함수입니다.
    # 이때 is_train에 따라 test set을 분리해서 load하도록 해야합니다.
    input_images = glob(os.path.join(self.dir_path, "image_2", "*.png"))
    label_images = glob(os.path.join(self.dir_path, "semantic", "*.png"))
    input_images.sort()
    label_images.sort()
    assert len(input_images) == len(label_images)
    data = [ _ for _ in zip(input_images, label_images)]

    if self.is_train:
      return data[:-30]
    return data[-30:]
    
  def __len__(self):
    # Generator의 length로서 전체 dataset을 batch_size로 나누고 소숫점 첫째자리에서 올림한 값을 반환합니다.
    return math.ceil(len(self.data) / self.batch_size)

  def __getitem__(self, index):
    # 입력과 출력을 만듭니다.
    # 입력은 resize및 augmentation이 적용된 input image이고 
    # 출력은 semantic label입니다.
    batch_data = self.data[
                           index*self.batch_size:
                           (index + 1)*self.batch_size
                           ]
    inputs = np.zeros([self.batch_size, *self.img_size])
    outputs = np.zeros([self.batch_size, *self.output_size])
        
    for i, data in enumerate(batch_data):
      input_img_path, output_path = data
      _input = imread(input_img_path)
      _output = imread(output_path)
      _output = (_output==7).astype(np.uint8)*1
      data = {
          "image": _input,
          "mask": _output,
          }
      augmented = self.augmentation(**data)
      inputs[i] = augmented["image"]/255
      outputs[i] = augmented["mask"]
      return inputs, outputs

  def on_epoch_end(self):
    # 한 epoch가 끝나면 실행되는 함수입니다. 학습중인 경우에 순서를 random shuffle하도록 적용한 것을 볼 수 있습니다.
    self.indexes = np.arange(len(self.data))
    if self.is_train == True :
      np.random.shuffle(self.indexes)
      return self.indexes
```
<br></br>
>
```python
augmentation = build_augmentation()
test_preproc = build_augmentation(is_train=False)
        
train_generator = KittiGenerator(
    dir_path, 
    augmentation=augmentation,
)

test_generator = KittiGenerator(
    dir_path, 
    augmentation=test_preproc,
    is_train=False
)
```
<br></br>

## 시맨틱 세그멘테이션 모델

### 모델 구조 만들기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/u-net_1kfpgqE.max-800x600.png)

U - Net 을 구현하며, 입력 이미지는 앞서 만든 데이터셋에 맞춰 만든다.

`Conv2D`, `UpSampling2D`, `MaxPooling2D`, `concatnate` 레이어 및 연산이 필요하며, `Dropout` 과 같은 기타 레이어를 적용할 수 있다.

<br></br>
> U - Net 모델 구현
```python
def build_model(input_shape=(224, 224, 3)):
    model = None
    # TODO: input_shape에 따라 U-Net을 만들어주세요
    # 이때 model은 fully convolutional해야 합니다.
   
    inputs = Input(input_shape)

      #Contracting Path
    conv1 = Conv2D(64, 3, activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool4)  
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv5)

      #Expanding Path
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(drop5)) 
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv9)  
    conv9 = Conv2D(2, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv9)     
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    return model
```
<br></br>

###  모델 학습

적절한 learning rate 와 epoch 를 찾아 모델을 학습하고 저장하자.

<br></br>
> 모델 학습 및 저장
```python
model = build_model()
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy')
model.fit_generator(
     generator=train_generator,
     validation_data=test_generator,
     steps_per_epoch=len(train_generator),
     epochs=100,
 )

model_path = dir_path + '/seg_model_unet.h5'
model.save(model_path)  #학습한 모델을 저장해 주세요.
```
<br></br>


## 시맨틱 세그멘테이션 모델 시각화

학습된 모델의 결과를 시각화를 통해 확인해보자.

테스트 데이터셋은 이미지만 제공할 뿐, label 데이터를 제공하지 않으므로 학습된 모델의 결과를 세그멘테이션 이미지로 만들어야 한다.

입력 이미지와 학습 모델의 출력 (결과) 를 쉽게 파악하기위해 오버레이 (overray) 를 통해 두 이미지를 겹쳐보자.

> 시맨틱 세그멘테이션 모델의 결과 시각화
> (저장된 모델은 `model = tf.keras.models.load_model(model_path)` 로 로드하여 활용할 수 있다.)
> (오버레이는 PIL 패키지의 `Image.blend` 를 활용)
```python
def get_output(model, preproc, image_path, output_path):
    # TODO: image_path로 입력된 이미지를 입력받아 preprocess를 해서 model로 infernece한 결과를 시각화하고 
    # 이를 output_path에 저장하는 함수를 작성해주세요.
     origin_img = imread(image_path)
     data = {"image":origin_img}
     processed = preproc(**data)
     output = model(np.expand_dims(processed["image"]/255,axis=0))
     output = (output[0].numpy()>0.5).astype(np.uint8).squeeze(-1)*255  #0.5라는 threshold를 변경하면 도로인식 결과범위가 달라집니다.
     output = Image.fromarray(output)
     background = Image.fromarray(origin_img).convert('RGBA')
     output = output.resize((origin_img.shape[1], origin_img.shape[0])).convert('RGBA')
     output = Image.blend(background, output, alpha=0.5)
     output.show()
     return output
 

# 완성한 뒤에는 시각화한 결과를 눈으로 확인해봅시다!
i = 1    # i값을 바꾸면 테스트용 파일이 달라집니다. 
get_output(
     model, 
     test_preproc,
     image_path=dir_path + f'/image_2/00{str(i).zfill(4)}_10.png',
     output_path=dir_path + f'./result_{str(i).zfill(3)}.png'
 )
```
세그멘테이션의 성능을 수치로 알아보기 위한 방법으로는 IoU (Intersection over Union) 이라는 척도를 사용한다.

IoU 을 구하기 위해서 모델이 도로라고 추론한 영역과 라벨 데이터에서 실제 도로 영역을 1, 나머지 영역을 0 으로 표시된 행렬이 필요하다.
<br></br>
> IoU 를 계산하는 함수를 생성
> 모델이 추론한 결과를 `prediction`, 실제 라벨 데이터를 `target` 라는 변수로 생성
```python
def calculate_iou_score(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = float(np.sum(intersection)) / float(np.sum(union))
    print('IoU : %f' % iou_score )
    return iou_score
```
<br></br>
> 앞서 생성한 `get_output` 을 확장하여 output. prediction, target 을 리턴하도록 구현
```python
def get_output(model, preproc, image_path, output_path, label_path):
    origin_img = imread(image_path)
    data = {"image":origin_img}
    processed = preproc(**data)
    output = model(np.expand_dims(processed["image"]/255,axis=0))
    output = (output[0].numpy()>=0.5).astype(np.uint8).squeeze(-1)*255  #0.5라는 threshold를 변경하면 도로인식 결과범위가 달라집니다.
    prediction = output/255   # 도로로 판단한 영역
    
    output = Image.fromarray(output)
    background = Image.fromarray(origin_img).convert('RGBA')
    output = output.resize((origin_img.shape[1], origin_img.shape[0])).convert('RGBA')
    output = Image.blend(background, output, alpha=0.5)
    output.show()   # 도로로 판단한 영역을 시각화!
     
    if label_path:   
        label_img = imread(label_path)
        label_data = {"image":label_img}
        label_processed = preproc(**label_data)
        label_processed = label_processed["image"]
        target = (label_processed == 7).astype(np.uint8)*1   # 라벨에서 도로로 기재된 영역

        return output, prediction, target
    else:
        return output, prediction, _
```
<br></br>
> IoU 를 통한 시각화
> 
```python
# 완성한 뒤에는 시각화한 결과를 눈으로 확인해봅시다!
i = 1    # i값을 바꾸면 테스트용 파일이 달라집니다. 
output, prediction, target = get_output(
     model, 
     test_preproc,
     image_path=dir_path + f'/image_2/00{str(i).zfill(4)}_10.png',
     output_path=dir_path + f'./result_{str(i).zfill(3)}.png',
     label_path=dir_path + f'/semantic/00{str(i).zfill(4)}_10.png'
 )

calculate_iou_score(target, prediction)
```
<br></br>
