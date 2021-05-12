# 18. 문자를 읽을 수 있는 딥러닝

## 학습 목표

1.  OCR의 과정을 이해합니다.
2.  문자인식 결과의 표현방식을 이해합니다.
3.  파이썬을 통해 OCR을 사용할 수 있습니다.

## 학습 준비

> 작업 디렉토리 설정
```bash
$ mkdir -p ~/aiffel/ocr_python
```
<br></br>

## 기계가 읽을 수 있나요?

문자 인식 기술은 현재 우편번호 추출을 통한 우편물 관리, 자동자 번호판 인식 등 실생활에서 널리 활용되고 있다. 이 문자 인식 기술은 OCR (Optical Character Recognition, 광학 문자 인식) 이라고 한다 . 즉, 기계가 문자를 읽는 것이다.

사람이 문자를 읽는 과정은 크게 문자를 인식하고, 인식한 문자를 해독하는 과정이다.

기계가 문자를 읽는 과정과 역시 사람과 다르지 않다. 문자를 인식 (Detection) 하고 어떤 문자인지 판독 (Recognition) 하는 과정이다.

구글에서는 클라우드 기반 OCR API 를 제공하고 있다.

+ 참고 : [구글 OCR API](https://cloud.google.com/vision/?utm_source=google&utm_medium=cpc&utm_campaign=japac-KR-all-en-dr-bkws-all-all-trial-e-dr-1008074&utm_content=text-ad-none-none-DEV_c-CRE_252596144846-ADGP_Hybrid+%7C+AW+SEM+%7C+BKWS+~+T1+%7C+EXA+%7C+ML+%7C+M:1+%7C+KR+%7C+en+%7C+Vision+%7C+API-KWID_43700029837773855-kwd-316837066534&userloc_1009877&utm_term=KW_google%20vision%20api%20ocr&gclid=Cj0KCQiAyp7yBRCwARIsABfQsnRMFOzgV84oX2MTWrPMvaE_JgjgTshUaLE6LYrsk8lM23-43gBfCkMaAnGaEALw_wcB)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-19-2.max-800x600_aTLyKtm.png)
<br></br>
위 링크에 들어가서 스크롤을 아래로 내리면 위 그림과 같은 화면을 만날 수 있으며, 여기에 사진을 드래그하면 쉽게 딥러닝 기반의 OCR 을 테스트 해볼 수 있다.

해당 링크에서 사용한 것은 `구글 OCR API` 를 이용한 것이다. 이 API 를 파이썬 코드로 다시 호출해 보자.

+ 구글 API 계정을 생성한 후 결제정보가 만료된 경우 OCR API 호출 시 billing 관련 오류가 발생할 수 있다. 결제계정을 재활성화해야 API 호출이 가능거나 새로운 계정을 생성하고 인증키를 다시 받는 방법이 있다.

<br></br>
> 터미널에 구글의 파이썬 API 인터페이스 모듈 설치
```python
pip install --upgrade google-api-python-client 

pip install google-cloud-vision
``` 
<br></br>
> Google Cloud Vision API 사용
```bash
$ cp ~/Downloads/sheet-contents-xxxx.json ~/aiffel/ocr_python/my_google_api_key.json
```
아래의 링크를 참고하여 서비스 계정 및 인증키를 생성하고, 브라우저에서 다운로드한 인증키는 위와 같은 경로에 `my_google_api_key.json`이라는 파일명으로 저장한다.

(파일은 처음에 sheet-contents-로 시작되는 이름으로 자동 저장된다.)

+ 참고 : [Google Cloud Vision API 사용하기](http://egloos.zum.com/mcchae/v/11342622)
<br></br>
> 인증키 경로 등록 후 커널 재시작
```bash
$ export GOOGLE_APPLICATION_CREDENTIALS=$HOME/aiffel/ocr_python/my_google_api_key.json
```
해당 명령어는 터미널에서 실행해 주어야 한다.
<br></br>
> 환경변수에 등록
```bash
$ echo  "export GOOGLE_APPLICATION_CREDENTIALS=$HOME/aiffel/ocr_python/my_google_api_key.json" >> ~/.bashrc
```
구글 API 를 계속 사용하고 싶다면 위와 같이 환경변수에 등록해준다.

환경변수에 등록해주면 매번 경로 변수 설정을 하지 않아도 된다.
<br></br>
> OCR API 구현
```python
def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()
        
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
       print('\n"{}"'.format(text.description))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                 for vertex in text.bounding_poly.vertices])

    print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
```
<br></br>
> OCR API 테스트
> (구글 OCR 링크에서 사용한 이미지를 재사용.) 
```python
# 다운받은 인증키 경로가 정확하게 지정되어 있어야 합니다. 
!ls -l $GOOGLE_APPLICATION_CREDENTIALS

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  os.getenv('HOME')+'/aiffel/ocr_python/my_google_api_key.json'

# 입력 이미지 경로를 지정해 주세요.
# (예시) path = os.getenv('HOME')+'/aiffel/ocr_python/test_image.png'
path = os.getenv('HOME')+'/aiffel/ocr_python/test_image.png'  

# 위에서 정의한 OCR API 이용 함수를 호출해 봅시다.
detect_text(path)
```
<br></br>

## 어떤 과정으로 읽을까요?

구글 API 에서는 문자의 영역을 사각형으로 표현하고 우측에 `Block`과 `Paragraph`로 구분해서 인식결과를 나타냈다.

구글 API 가 이미지에 박스를 친 다음 박스별 텍스트의 내용을 알려준 것처럼, 문자 모델은 보통 두 단계로 이뤄진다.

먼저 입력받은 사진 속에서 문자의 위치를 찾아내는 과정을 Text Detection (문자검출) 이라고 한다.

이렇게 찾은 문자 영역으로부터 문자를 읽어내는 것을 Text Recognition (문자인식) 이라고 한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-19-4.max-800x600_lhFXcl2.png)
<br></br>
위 그림은 카카오의 OCR 모델이다. 그림을 보면 모델은 먼저 문자가 있는 영역의 정보 (`coord, Text Recognition`) 를 찾아내고, 각 영역에서 문자를 인식하고 있다. 이렇게 문자 인식 모델은 Text Detection 과 Recognition 두 과정을 통해서 사진 속의 문자를 읽을 수 있다.

그림을 보면 문자 영역을 표현하는 방법으로 사각형의 네 꼭지점 좌표를 알려주는 방법을 제시한다.

+ 참고 : [논문 : Scene Text Detection with Polygon Offsetting and Border Augmentation](https://www.mdpi.com/2079-9292/9/1/117/pdf)
<br></br>

## 딥러닝 문자인식의 시작

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-19-5.max-800x600_fSsqPX1.png)
<br></br>
위 그림은 LeNet - 5 의 구조이다. 문자 인식의 모델 중 하나로 간단하게 Convolution 레이어와 최종 출력 레이어로 이루어져 있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-19-6.gif)
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-19-7.png)
<br></br>
위 그림은 입력과 각 레이어의 활성화를 시각화한 LeNet 의 MNIST 데모 이미지이다.

아래 데모사이트에서 이렇게 간단한 구조로도 어려운 글자를 읽을 수 있는 딥러닝 분류 모델, LeNet 을 확인해 보자.

+ 참고 : 
	+ [Yann LeCun's Demo](http://yann.lecun.com/exdb/lenet/stroke-width.html)
	+ [L12/1 LeNet from Berkely](https://youtu.be/m3BrTjo2zUA)
<br></br>
하지만 이렇게 단순한 분류 모델만으로는 넓고 복잡한 이미지에서 글자 영역을 찾을 수 없을 뿐더러 글자를 영역별로 잘라서 넣더라도 우리가 인식하기를 원하는 사진은 여러 글자가 모여있기 때문에 단순한 분류 문제로 표현이 불가능하기 때문에 구글 API 로 테스트해 보았던 복잡한 결과를 얻을 수 없다.
<br></br>

## 사진 속 문자 찾아내기 - Detection

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-19-9.max-800x600_lt28Nt5.jpg)
![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-19-8.max-800x600_wGIgZfi.jpg)
<br></br>
사진 속 문자를 찾아내는 딥러닝 모델은 일바적으로 Object Detection (객체 인식) 방법으로 접근한다.

문자는 일반적인 객체가 어떤 물체인지에 따라 크기가 일정한 반면에 영역과 배치가 자유롭다. 

또한 일반적으로 객체는 물체간 거리가 어느정도 확보되는데 반해 글자는 물체간 거리가 비교적 매우 촘촘하게 배치되어있다. 따라서 문자의 특성에 따라 모델을 변경하기도 한다.

딥러닝 기반 Object Detection 방법에는 Regression (회귀), Segmentation (세그멘테이션) 방식 2 가지로 나뉘며, 회귀를 기준으로 하는 방법은 기준으로 하는 박스 대비 문자가 얼마나 차이가 나는지를 학습한다.

세그멘테이션을 기준으로 하는 방법은 픽셀단위로 해당 픽셀이 문자를 표현하는지 분류하는 문제 (Pixel - Wise Classification) 이라고 볼 수 있다. 

+ 참고 : [딥러닝을 활용한 객체 탐지 알고리즘 이해하기](https://blogs.sas.com/content/saskorea/2018/12/21/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-%EA%B0%9D%EC%B2%B4-%ED%83%90%EC%A7%80-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0/)
<br></br>

## 사진 속 문자 읽어내기 - Recognition

문자 인식 (Text Recognition) 은 사진 속에서 문자를 검출하는 검출 모델이 영역을 잘라서 주면 그 영역에 어떤 글자가 포함되어 있는지를 읽어내는 과정을 의미한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-19-12.max-800x600_y9DP2Mu.png)<br></br>
위 그림은 ICDAR 15 라는 OCR 데이터셋에서 단어 단위로 잘린 이미지를 나타낸 것이다.

문자 인식 모델은 위 그림과 같이 Object Detection 을 통해 작제 잘린 이미지를 입력받아 이미지 속에 어떤 단어가 포함되었는지 찾아낸다.

해당 방법은 이미지 분류 문제보다 자연어 처리에서 크게 영감을 받은 모델이다.

대표적인 자연어처리의 모델은 RNN 이 있는데, 문자 인식 모델의 기본적인 방법 중 하나는 CNN 과 RNN 을 결합한 CRNN 모델을 사용하는 것이다.

즉, 이미지 내의 텍스트와 관련있는 특징을 CNN 으로 추출한 후 스텝 단위의 문자 정보를 RNN 으로 인식하는 것이다.

+ 참고 :
	+ [네이버 데뷰 2018, 이활석님의 CRAFT 모델소개와 연구 경험](https://tv.naver.com/v/4578167)
	+ [Terry TaeWoong Um님의 사진 속 글자 읽기, OCR (Optical character recognition)](https://www.youtube.com/watch?v=ckRFBl_XWFg)
<br></br>

## Keras - OCR 써보기

앞서 구글 API 를 통해 체험해본 OCR 모델은 텐서플로우를 기반으로 모델 구현이 가능하다. `keras-ocr` 은 텐서플로우의 케라스 API 를 기반으로 이미지속 문자를 읽어내는 End - to - End OCR 을 구현할 수 있게 도와준다.

(`keras-ocr` 은 tensorflow 2.2.0 에서 구동된다. 해당 버전 이상의 텐서플로우에서는 미리 학습된 모델에서 오류가 발생할 수 있다.)
<br></br>
> 텐서플로우 버전 확인
```python
pip list | grep tensorflow 

# 만약 tensorflow 버전이 맞지 않다면 재설치를 해줍시다. 
pip uninstall tensorflow 
pip install tensorflow==2.2.0
```
<br></br>
> `keras-ocr` 라이브러리 설치
```python
pip install keras-ocr
```
<br></br>
> GPU 사용시 cuDNN 관련 에러가 발생했다면?
```bash
echo  $TF_FORCE_GPU_ALLOW_GROWTH
```
터미널에서 위 명령어를 실행했을 때 True 가 출력될 경우 환경설정에 반영되어 있는 것이다.

 환경 설정이 반영되어 있지 않으면 코드 구동 과정에서 OOM (Out Of Memory) 에러가 발생할 수 있다.

에러가 발생할 경우 터미널에서 아래 명령어를 사용한다.
<br></br>
> True 가 출력되지 않을 경우
```bash
$ echo  "export TF_FORCE_GPU_ALLOW_GROWTH=true" >> ~/.bashrc
```
<br></br>
> `keras-ocr` 의 인식결과를 위한 시각화 라이브러리 가져오기
```python
import matplotlib.pyplot as plt
import keras_ocr

# keras-ocr이 detector과 recognizer를 위한 모델을 자동으로 다운로드받게 됩니다. 
pipeline = keras_ocr.pipeline.Pipeline()
```
`keras_ocr.pipeline.Pipeline()` 는 인식을 위한 파이프라인을 생성하는데 이때 초기화 과정에서 미리 학습된 모델의 가중치(weight)를 불러오게 된다.

즉, 검출기와 인식기를 위한 가중치를 하나씩 불러온다.
<br></br>
> 파이프라인의 `recognize()` 에 이미지 추가
```python
# 테스트에 사용할 이미지 url을 모아 봅니다. 추가로 더 모아볼 수도 있습니다. 
image_urls = [
  'https://source.unsplash.com/M7mu6jXlcns/640x460',
  'https://source.unsplash.com/6jsp4iHc8hI/640x460',
  'https://source.unsplash.com/98uYQ-KupiE',
  'https://source.unsplash.com/j9JoYpaJH3A',
  'https://source.unsplash.com/eBkEJ9cH5b4'
]

images = [ keras_ocr.tools.read(url) for url in image_urls]
prediction_groups = [pipeline.recognize([url]) for url in image_urls]
```
이미지 url 소르를 통해 추가해준다.

+ [이미지 출처](https://unsplash.com/s/photos/text)
<br></br>
> 인식된 결과를 시각화
```python
# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for idx, ax in enumerate(axs):
    keras_ocr.tools.drawAnnotations(image=images[idx], 
                                    predictions=prediction_groups[idx][0], ax=ax)
```
내부적으로 `recognize()` 는 검출기와 인식기를 두고, 검출기로 바운딩 박스(bounding box, 문자가 있는 영역을 표시한 정보)를 검출한 뒤, 인식기가 각 박스로부터 문자를 인식하는 과정을 거치도록 한다.

+ 참고 : [keras-ocr 파이프라인](https://github.com/faustomorales/keras-ocr/blob/master/keras_ocr/pipeline.py)
<br></br>

+ keras-ocr 은 한글 데이터셋으로 훈련되어 있지 않은 모델이므로, Text Detection 이 정상적으로 작동하더라고 Recognition 의 결과가 잘 못 나올 수 있다.

+ 참고 : [keras-ocr github](https://github.com/faustomorales/keras-ocr/issues/101)
<br></br>

## 테서랙트 써보기

이번에는 구글에서 후원하는 OCR 오픈소스 라이브러리인 테서렉트를 통해 문자 인식을 해보자.

오픈소스라는 점에서 프로젝트 틍에 활용하기 쉽다는 장점이 있으며, 현재 버전 4와 Tessearct.js 등으로 확장되는 등 많은 곳에서 사용되고 있으며, 한국어를 포함한 116 개 국어를 지원한다.

<br></br>
> 테서랙트 설치
```bash
$ sudo apt install tesseract-ocr 
$ sudo apt install libtesseract-dev
```
+ 참고 : [운영체제에 따른 Tesseract Install Guide](https://github.com/tesseract-ocr/tesseract/wiki)
<br></br>
> 테서렉트 파이썬 wrapper  설치
```python
pip install pytesseract
```
`Pytesseract`는 OS에 설치된 테서랙트를 파이썬에서 쉽게 사용할 수있도록 해주는 래퍼 라이브러리(wrapper library) 이며, 파이썬 내에서 컴퓨터에 설치된 테서랙트 엔진의 기능을 바로 쓸 수 있도록 해준다.

+ 참고 : 
	+ [ytesseract](https://pypi.org/project/pytesseract/)
	+ [위키백과: 래퍼 라이브러리](https://ko.wikipedia.org/wiki/%EB%9E%98%ED%8D%BC_%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC)
<br></br>
> 태서렉트로 문자 검출하고 이미지 검출하는 함수 생성
```python
import os
import pytesseract
from PIL import Image
from pytesseract import Output
import matplotlib.pyplot as plt

# OCR Engine modes(–oem):
# 0 - Legacy engine only.
# 1 - Neural nets LSTM engine only.
# 2 - Legacy + LSTM engines.
# 3 - Default, based on what is available.

# Page segmentation modes(–psm):
# 0 - Orientation and script detection (OSD) only.
# 1 - Automatic page segmentation with OSD.
# 2 - Automatic page segmentation, but no OSD, or OCR.
# 3 - Fully automatic page segmentation, but no OSD. (Default)
# 4 - Assume a single column of text of variable sizes.
# 5 - Assume a single uniform block of vertically aligned text.
# 6 - Assume a single uniform block of text.
# 7 - Treat the image as a single text line.
# 8 - Treat the image as a single word.
# 9 - Treat the image as a single word in a circle.
# 10 - Treat the image as a single character.
# 11 - Sparse text. Find as much text as possible in no particular order.
# 12 - Sparse text with OSD.
# 13 - Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

def crop_word_regions(image_path='./images/sample.png', output_path='./output'):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    custom_oem_psm_config = r'--oem 3 --psm 3'
    image = Image.open(image_path)

    recognized_data = pytesseract.image_to_data(
        image, lang='eng',    # 한국어라면 lang='kor'
        config=custom_oem_psm_config,
        output_type=Output.DICT
    )
    
    top_level = max(recognized_data['level'])
    index = 0
    cropped_image_path_list = []
    for i in range(len(recognized_data['level'])):
        level = recognized_data['level'][i]
    
        if level == top_level:
            left = recognized_data['left'][i]
            top = recognized_data['top'][i]
            width = recognized_data['width'][i]
            height = recognized_data['height'][i]
            
            output_img_path = os.path.join(output_path, f"{str(index).zfill(4)}.png")
            print(output_img_path)
            cropped_image = image.crop((
                left,
                top,
                left+width,
                top+height
            ))
            cropped_image.save(output_img_path)
            cropped_image_path_list.append(output_img_path)
            index += 1
    return cropped_image_path_list


work_dir = os.getenv('HOME')+'/aiffel/ocr_python'
img_file_path = work_dir + '/test_image.png'   #테스트용 이미지 경로입니다. 본인이 선택한 파일명으로 바꿔주세요. 

cropped_image_path_list = crop_word_regions(img_file_path, work_dir)
```
테서랙트를 사용하면 한 번에 이미지 내의 문자 검출과 인식을 할 수 있다.

`crop_word_regions()` 함수는 여러분이 선택한 테스트 이미지를 받아서, 문자 검출을 진행한 후, 검출된 문자 영역을 crop 한 이미지로 만들어 그 파일들의 list를 리턴하는 함수이다.

기본적으로 `pytesseract.image_to_data()` 를 사용하며 편하게 사용하기 위해서 `pytesseract` 의 `Output` 을 사용해서 결과값의 형식을 딕셔너리(`DICT`) 형식으로 설정해준다.

인식된 결과는 바운딩 박스의 left, top, width, height 정보를 가지게되며, 바운딩 박스를 사용해 이미지의 문자 영역들을 `PIL(pillow)` 또는 `opencv` 라이브러리를 사용해 잘라 (crop) 서 `cropped_image_path_list` 에 담아 리턴한다.

+ lang='kor' 로 바꾸면 에러가 발생하며, 테서랙트의 언어팩을 설치해야 정상동작한다.
<br></br>
> 테서렉트의 언어팩 설치
```bash
$ sudo apt install tesseract-ocr-kor
```
+ 참고 : [테서렉트 언어팩 설치](http://blog.daum.net/rayolla/1141)
<br></br>
> 테서랙트로 잘인 이미지에서 단어 인식을 수행하는 함수 생성
```python
def recognize_images(cropped_image_path_list):
    custom_oem_psm_config = r'--oem 3 --psm 7'
    
    for image_path in cropped_image_path_list:
        image = Image.open(image_path)
        recognized_data = pytesseract.image_to_string(
            image, lang='eng',    # 한국어라면 lang='kor'
            config=custom_oem_psm_config,
            output_type=Output.DICT
        )
        print(recognized_data['text'])
    print("Done")

# 위에서 준비한 문자 영역 파일들을 인식하여 얻어진 텍스트를 출력합니다.
recognize_images(cropped_image_path_list)
```
검출된 바운딩 박스 별로 잘린 이미지를 넣어주면 영역별 텍스트가 결과값으로 나오는 `image_to_string()`를 사용한다.
<br></br>
